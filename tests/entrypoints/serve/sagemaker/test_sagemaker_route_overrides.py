"""SageMaker handler overrides and LoRA routes through a real FastAPI/ASGI stack.

Runs on the staged vllm package. No model is loaded: the engine and serving
objects are minimal stand-ins at the app.state boundary vLLM itself uses.
Regression for FastAPI >= 0.137, where include_router() stores nested
_IncludedRouter nodes and the container standards' route-conflict scan stopped
seeing vLLM's own /ping and /invocations (overrides were silently ignored).
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

import httpx
from fastapi import FastAPI, Request
from model_hosting_container_standards.common.handler import handler_registry
from starlette.datastructures import State

from vllm.entrypoints.serve.engine.protocol import ErrorInfo, ErrorResponse
from vllm.entrypoints.serve.lora.api_router import attach_router as attach_lora
from vllm.entrypoints.serve.sagemaker.api_router import (
    attach_router as attach_sagemaker,
)
from vllm.entrypoints.serve.sagemaker.api_router import sagemaker_standards_bootstrap

HANDLERS = '''
from fastapi import Request
from fastapi.responses import JSONResponse

async def custom_ping(raw_request: Request):
    return JSONResponse({"custom": "ping"})

async def custom_invocation(raw_request: Request):
    return JSONResponse({"custom": "invocation", "echo": await raw_request.json()})
'''


class _Engine:
    async def check_health(self):
        return None


class _Tokenization:
    def create_error_response(self, message, **kwargs):
        return ErrorResponse(error=ErrorInfo(message=message, type="BadRequestError", code=400))


class _Chat:
    def __init__(self):
        self.contents = []

    async def create_chat_completion(self, request, raw_request):
        self.contents.append(request.messages[0]["content"])
        return ErrorResponse(error=ErrorInfo(message="chat-handler-reached", type="BadRequestError", code=400))


class _Models:
    def __init__(self):
        self.loaded = []
        self.unloaded = []

    async def load_lora_adapter(self, request):
        self.loaded.append((request.lora_name, request.lora_path))
        return "loaded"

    async def unload_lora_adapter(self, request):
        self.unloaded.append(request.lora_name)
        return "unloaded"


_STATE_GETATTR = State.__getattr__


def _unset_serving_attribute(self, key):
    # vLLM sets every serving object (None when unsupported); mirror that.
    try:
        return _STATE_GETATTR(self, key)
    except AttributeError:
        if key.startswith(("openai_serving", "serving_", "anthropic_serving")):
            return None
        raise


def build_app():
    handler_registry.clear()
    app = FastAPI()
    app.state.engine_client = _Engine()
    app.state.serving_tokenization = _Tokenization()
    app.state.openai_serving_chat = _Chat()
    app.state.openai_serving_models = _Models()
    attach_sagemaker(app, supported_tasks=("generate",))
    attach_lora(app)
    sagemaker_standards_bootstrap(app)
    return app


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


class SageMakerRouter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = mock.patch.object(State, "__getattr__", _unset_serving_attribute, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ, {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1"})
        env.start()
        self.addCleanup(env.stop)
        for key in ("CUSTOM_FASTAPI_PING_HANDLER", "CUSTOM_FASTAPI_INVOCATION_HANDLER"):
            os.environ.pop(key, None)

    async def test_default_ping_and_invocation_dispatch(self):
        app = build_app()
        async with client(app) as http:
            self.assertEqual((await http.get("/ping")).status_code, 200)
            bad = await http.post("/invocations", content=b"{broken", headers={"content-type": "application/json"})
            self.assertEqual(bad.status_code, 400)
            nomatch = await http.post("/invocations", json={"unknown": "shape"})
            self.assertEqual(nomatch.status_code, 400)
            self.assertIn("Cannot find suitable handler", nomatch.text)
            chat = await http.post(
                "/invocations", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]}
            )
            self.assertIn("chat-handler-reached", chat.text)
            self.assertEqual(app.state.openai_serving_chat.contents, ["hi"])

    async def test_environment_handler_overrides_replace_vllms_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "sagemaker_overrides.py"), "w") as handle:
                handle.write(HANDLERS)
            with mock.patch.object(sys, "path", [tmp, *sys.path]), mock.patch.dict(
                os.environ,
                {
                    "CUSTOM_FASTAPI_PING_HANDLER": "sagemaker_overrides:custom_ping",
                    "CUSTOM_FASTAPI_INVOCATION_HANDLER": "sagemaker_overrides:custom_invocation",
                },
            ):
                app = build_app()
                async with client(app) as http:
                    ping = await http.get("/ping")
                    self.assertEqual(ping.json(), {"custom": "ping"})
                    invoked = await http.post("/invocations", json={"a": 1})
                    self.assertEqual(invoked.json(), {"custom": "invocation", "echo": {"a": 1}})
                    # The override replaced the route; vLLM's own was not left behind it.
                    self.assertEqual(
                        sum(1 for r in app.router.routes if getattr(r, "path", None) == "/ping"), 1
                    )


class LoraRouter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = mock.patch.object(State, "__getattr__", _unset_serving_attribute, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ, {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1"})
        env.start()
        self.addCleanup(env.stop)

    async def test_load_and_unload_validate_and_forward(self):
        app = build_app()
        models = app.state.openai_serving_models
        async with client(app) as http:
            self.assertEqual((await http.post("/v1/load_lora_adapter", json={})).status_code, 422)
            self.assertEqual((await http.post("/v1/unload_lora_adapter", json={})).status_code, 422)
            loaded = await http.post(
                "/v1/load_lora_adapter", json={"lora_name": "a1", "lora_path": "/nonexistent/adapter"}
            )
            self.assertEqual(loaded.status_code, 200)
            unloaded = await http.post("/v1/unload_lora_adapter", json={"lora_name": "a1"})
            self.assertEqual(unloaded.status_code, 200)
        self.assertEqual(models.loaded, [("a1", "/nonexistent/adapter")])
        self.assertEqual(models.unloaded, ["a1"])


if __name__ == "__main__":
    unittest.main()
