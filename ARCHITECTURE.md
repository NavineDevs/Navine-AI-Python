# Architecture (Phase 1 hybrid accelerators)

## Reality

Navine AI (twin of Navuryx AI) is a **Python/PyTorch monorepo** with optional native accelerators.

| Layer | Role |
|-------|------|
| `navine/` | Control plane: train, chat, image, video, API (FastAPI) |
| `native/rust/navops` | Optional PyO3 SIMD helpers + KV cache container |
| `native/cpp/navcuda` | Optional C++ DDIM schedule helpers (CPU; CUDA when built) |
| `services/gateway` | Go reverse proxy + rate limit in front of FastAPI |
| `deploy/docker` | API + gateway containers |
| `app/src-tauri` | Desktop shell only (not model runtime) |

**Not in Phase 1:** Java/Spring, Zig, multi-region K8s, gRPC workers, shared-memory multi-process train, FlashAttention full port.

## Data flow

```
Client -> Go gateway (:8080) -> FastAPI (:8765) -> PyTorch models
                                      |-> optional navops / navcuda
```

## Custom-only models

Text (`NavineTextModel`) and image (`NavineDiffusionModel` / SimpleUNet) are trained locally. No Hugging Face base models are required for core paths. Unrestricted research mode remains enabled in product config; no content moderation layer is added in this architecture.

## Install natives (optional)

```bat
rustup default stable
pip install maturin
cd native\rust\navops && maturin develop --release

cd native\cpp\navcuda && pip install -e .
```

Without natives, pure Python paths run automatically via `navine.native_bridge`.

## Gateway

```bat
cd services\gateway
go run . -upstream http://127.0.0.1:8765 -listen :8080
```

## Docker

```bat
cd deploy\docker
docker compose up --build
```

## Mirror

Navine receives the same layout with package renames via `scripts/sync_native_stack.py`.
