# Local patches (`local` branch of eejd/py-vllm)

`local` = upstream release tag + the `topic/*` branches below. `main` is a fast-forward mirror of
`vllm-project/vllm` `main` and is never edited. Release tags: `v<base>-local<NN>`.
Convention: eejd/macports-ports-local `docs/development-model.md` ("Forks, upstreams and port tiers").

| Topic branch | Behaviour | Upstream status |
|---|---|---|
| `topic/sagemaker-route-conflicts` | SageMaker `/ping` and `/invocations` are registered directly on the app router instead of through `app.include_router()`, so `CUSTOM_FASTAPI_PING_HANDLER` / `CUSTOM_FASTAPI_INVOCATION_HANDLER` overrides keep working on FastAPI >= 0.137 (model-hosting-container-standards' route-conflict scan only sees `APIRoute` entries; `include_router()` stores a nested `_IncludedRouter` there since FastAPI 0.137). Test: `tests/entrypoints/serve/sagemaker/test_sagemaker_route_overrides.py`. | Not submitted. Upstream instead caps `fastapi < 0.137.0` in `requirements/common.txt` (vllm-project/vllm#45594, closed; #45629 merged for a different symptom). The MacPorts build relaxes that cap (ships FastAPI 0.142), so this patch is what keeps the overrides working there. Retire when upstream lifts the cap or fixes the scan. |

MacPorts packaging adaptations (dependency bounds relaxed to what MacPorts ships, dropped optional
deps) are NOT carried here: they stay `patchfiles` in the port (`py-vllm-local`).
