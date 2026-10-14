import argparse

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.staticfiles import StaticFiles

from navine import __version__
from navine.api.game_routes import game_router
from navine.api.moltbook_routes import moltbook_router
from navine.api.ollama_routes import ollama_router
from navine.api.openai_routes import openai_router
from navine.api.routes import router
from navine.api.settings import get_key_store, load_api_config
from navine.auth.middleware import AuthMiddleware, RateLimitMiddleware
from navine.policy import enforce_local_only
from navine.utils.brand import brand_name, brand_port, load_brand
from navine.utils.paths import get_project_root

enforce_local_only()

_api_config = load_api_config()
_js_path = get_project_root() / "web" / "assets" / "app.js"
_ASSET_VERSION = (
    f"{__version__.replace('.', '')}-{int(_js_path.stat().st_mtime)}"
    if _js_path.exists()
    else __version__.replace(".", "")
)

WEB_PAGE_PATHS = (
    "/",
    "/chat",
    "/voice",
    "/video-chat",
    "/image",
    "/generate",
    "/osint",
    "/deepfake",
    "/clone",
    "/music",
    "/info",
    "/train",
    "/api-gen",
    "/settings",
)


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
        return response


class AssetVersionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.endswith(".html"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
        return response


def create_app() -> FastAPI:
    brand = load_brand()
    product = brand_name()
    app = FastAPI(
        title=product,
        description=f"{product} local multimodal API. 100% local PyTorch inference. No external AI APIs.",
        version=__version__,
        docs_url=None,
        redoc_url=None,
    )
    cors_origins = _api_config.get("cors_origins") or ["*"]
    key_store = get_key_store(_api_config)
    app.add_middleware(AssetVersionMiddleware)
    app.add_middleware(AuthMiddleware, api_config=_api_config, key_store=key_store)
    app.add_middleware(RateLimitMiddleware, api_config=_api_config)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api")
    app.include_router(game_router, prefix="/api")
    app.include_router(moltbook_router, prefix="/api")
    app.include_router(openai_router, prefix="/v1")
    app.include_router(ollama_router, prefix="/ollama/api")
    web_dir = get_project_root() / "web"
    if web_dir.is_dir():
        app.mount("/assets", NoCacheStaticFiles(directory=web_dir / "assets"), name="assets")
        downloads = web_dir / "downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        app.mount("/downloads", NoCacheStaticFiles(directory=downloads), name="downloads")

        def _render_index():
            html = (web_dir / "index.html").read_text(encoding="utf-8")
            html = html.replace("{{ASSET_VERSION}}", _ASSET_VERSION)
            html = html.replace("{{BRAND_NAME}}", product)
            html = html.replace("{{ENGINE_NAME}}", str(brand.get("engine_name") or product))
            from fastapi.responses import HTMLResponse

            return HTMLResponse(
                html,
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                },
            )

        for page_path in WEB_PAGE_PATHS:
            app.add_api_route(page_path, _render_index, methods=["GET"], include_in_schema=False)

        @app.get("/docs")
        def api_docs():
            docs_path = web_dir / "api-docs.html"
            if docs_path.is_file():
                return FileResponse(docs_path)
            raise HTTPException(status_code=404, detail="API documentation not found")

    @app.on_event("startup")
    async def ensure_api_key():
        try:
            from navine.downloads import ensure_download_packages

            ensure_download_packages()
        except Exception:
            pass
        try:
            from navine.moltbook import start_heartbeat

            start_heartbeat()
        except Exception:
            pass
        raw_key = key_store.ensure_default_key()
        if raw_key:
            print(f"{product}: no API keys found. Generated default key (shown once):")
            print(raw_key)
            print(f"Store location: {key_store.path}")

    return app


app = create_app()


def main():
    parser = argparse.ArgumentParser(description=f"{brand_name()} local web server")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    host = args.host or _api_config.get("bind_host", "127.0.0.1")
    port = args.port or int(_api_config.get("port") or brand_port())
    print(f"{brand_name()} server v{__version__} starting on http://{host}:{port}")
    uvicorn.run("navine.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
