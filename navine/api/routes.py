import base64
import json
import re
from pathlib import Path
from typing import Optional
import torch
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from navine.api.chat_utils import normalize_chat_request
from navine.api.schemas import (
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    ApiKeyInfo,
    ApiKeyListResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    CodeRequest,
    CodeResponse,
    CodeRunRequest,
    CodeRunResponse,
    DeepfakeFaceRequest,
    DeepfakeVideoRequest,
    OsintInvestigateRequest,
    OsintInvestigateResponse,
    DesktopInputRequest,
    DesktopSettingsRequest,
    DesktopStatusResponse,
    FileCreateRequest,
    FileCreateResponse,
    ZipCreateRequest,
    McpServerUpsertRequest,
    McpServerRemoveResponse,
    HealthResponse,
    ImageRequest,
    ImageResponse,
    InfoResponse,
    ModelStatus,
    ProductProgress,
    ScreenCaptureRequest,
    ScreenDescribeRequest,
    TextRequest,
    TextResponse,
    TrainingTypeInfo,
    TrainingTypesResponse,
    AutolearnStatusResponse,
    MarathonStatusResponse,
    VideoRequest,
    VideoResponse,
    VoiceCloneRequest,
    VoiceSpeakRequest,
    VoiceTranscribeRequest,
    MusicRequest,
    MusicResponse,
    CameraDescribeRequest,
    TrainProgressResponse,
    ModelsStatsResponse,
    AdminUsernameRequest,
    AdminLoginRequest,
    AdminLoginResponse,
    AdminSessionResponse,
    AuthSignupRequest,
    AuthLoginRequest,
    AuthLoginResponse,
    AuthSessionResponse,
    UserChatSummary,
    UserChatListResponse,
    UserChatCreateRequest,
    UserChatRenameRequest,
    UserChatMessage,
    UserChatDetailResponse,
    UserSettings,
    UserSettingsResponse,
    TrainStartRequest,
    TrainCustomRequest,
    TrainJobResponse,
    TrainJobsResponse,
    TrainLearnRequest,
)
from navine.utils.paths import get_checkpoint_dir
from navine.policy import is_local_only
from navine.utils.brand import (
    brand_backend,
    brand_name,
    brand_ui_features,
    brand_ui_tabs,
    filter_foreign_brand,
    is_foreign_brand_text,
    is_python_only,
    load_brand,
    python_train_denied,
    scrub_cross_brand_text,
)

router = APIRouter()


def _session_token_from_request(request: Request) -> Optional[str]:
    from navine.auth.admin import admin_token_from_headers
    from navine.auth.users import session_from_headers

    return session_from_headers(request.headers) or admin_token_from_headers(request.headers)


def _session_user(request: Request) -> Optional[str]:
    from navine.auth.users import verify_session_token

    return verify_session_token(_session_token_from_request(request))


def _require_user(request: Request) -> str:
    user = _session_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in required")
    return user


def _require_admin(request: Request) -> str:
    from navine.auth.users import UserStore

    user = _session_user(request)
    if not user or not UserStore().is_admin(user):
        raise HTTPException(status_code=401, detail="Admin login required")
    return user


def _apply_chat_image_attachment(user_message: str, req: ChatRequest) -> str:
    image_b64 = getattr(req, "image_base64", None)
    if not image_b64:
        return user_message
    try:
        from navine.desktop.vision import describe_image_bytes

        raw_b64 = str(image_b64 or "")
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        raw = base64.b64decode(raw_b64)
        described = describe_image_bytes(
            raw,
            question=user_message or "Describe this image and answer any question about it.",
            use_model=True,
            mime=str(getattr(req, "image_mime", None) or "image/jpeg"),
            label="chat_attach",
        )
        vision_text = str(
            described.get("text")
            or described.get("summary")
            or described.get("analysis")
            or ""
        ).strip()
        if vision_text:
            if user_message:
                return f"{user_message}\n\n[Attached image analysis]\n{vision_text}"
            return f"Please discuss this attached image.\n\n[Attached image analysis]\n{vision_text}"
        if not user_message:
            raise HTTPException(status_code=400, detail="Could not read attached image")
        return user_message
    except HTTPException:
        raise
    except Exception as exc:
        if not user_message:
            raise HTTPException(status_code=400, detail=f"Image attach failed: {exc}") from exc
        return f"{user_message}\n\n[Attached image could not be analyzed: {exc}]"


def _require_train_feature() -> None:
    if is_python_only() or not brand_ui_features().get("train", True):
        raise HTTPException(status_code=403, detail="Training is disabled on this site")


def _token_expires(token: Optional[str]) -> Optional[int]:
    if not token or "." not in token:
        return None
    try:
        payload = token.rsplit(".", 1)[0]
        return int(json.loads(payload).get("exp") or 0)
    except Exception:
        return None


def _resolve_train_media(req) -> dict:
    from navine.utils.media_upload import resolve_image_input, save_upload_audio

    out: dict = {}
    voice_sample = getattr(req, "voice_sample", None)
    if getattr(req, "voice_sample_base64", None):
        path = save_upload_audio(req.voice_sample_base64, prefix="train_voice")
        out["voice_sample"] = str(path)
    elif voice_sample:
        out["voice_sample"] = str(voice_sample)

    deepfake_source = getattr(req, "deepfake_source", None)
    deepfake_target = getattr(req, "deepfake_target", None)
    if getattr(req, "deepfake_source_base64", None):
        out["deepfake_source"] = str(
            resolve_image_input(data_base64=req.deepfake_source_base64, label="deepfake_source")
        )
    elif deepfake_source:
        out["deepfake_source"] = str(deepfake_source)
    if getattr(req, "deepfake_target_base64", None):
        out["deepfake_target"] = str(
            resolve_image_input(data_base64=req.deepfake_target_base64, label="deepfake_target")
        )
    elif deepfake_target:
        out["deepfake_target"] = str(deepfake_target)
    return out


STAGE_CATALOG = [
    "multimodal-text",
    "nsfw",
    "unrestricted",
    "general",
    "creative",
    "coding",
    "math",
    "chat",
    "thinking",
    "detective",
    "osint",
    "games",
    "hitboyx23",
    "hitboyx23_python",
    "image",
    "video",
    "voice",
    "eval",
]


def _model_status(name: str) -> ModelStatus:
    from datetime import datetime, timezone

    from navine.utils.training_lock import training_lock_active

    if name == "voice":
        ckpt = get_checkpoint_dir("voice") / "train_report.json"
        if not ckpt.exists():
            ckpt = get_checkpoint_dir("voice") / "latest.pt"
    elif name in (
        "text_enterprise",
        "text_code",
        "hitboyx23_ai",
        "hitboyx23_ai_python",
        "image_enterprise",
        "video_enterprise",
    ):
        try:
            from navine.utils.tier import load_modality_config, resolve_checkpoint_dir

            modality = "image" if name.startswith("image") else ("video" if name.startswith("video") else "text")
            if name in ("text_code", "hitboyx23_ai", "hitboyx23_ai_python"):
                ckpt = get_checkpoint_dir(name) / "latest.pt"
            else:
                cfg = load_modality_config(modality)
                ckpt = resolve_checkpoint_dir(modality, cfg) / "latest.pt"
                if not ckpt.exists():
                    ckpt = get_checkpoint_dir(name) / "latest.pt"
        except Exception:
            ckpt = get_checkpoint_dir(name) / "latest.pt"
    else:
        try:
            from navine.utils.tier import load_modality_config, resolve_checkpoint_dir

            cfg = load_modality_config(name)
            ckpt = resolve_checkpoint_dir(name, cfg) / "latest.pt"
            if not ckpt.exists():
                ckpt = get_checkpoint_dir(name) / "latest.pt"
        except Exception:
            ckpt = get_checkpoint_dir(name) / "latest.pt"
    mtime = None
    age = None
    if ckpt.exists():
        ts = ckpt.stat().st_mtime
        mtime = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        age = max(0.0, datetime.now(timezone.utc).timestamp() - ts)
    return ModelStatus(
        name=name,
        trained=ckpt.exists(),
        checkpoint=str(ckpt),
        mtime=mtime,
        age_seconds=age,
        training=bool(training_lock_active(name if name != "voice" else "voice")),
    )


def _parse_master_live(recent_or_all: list[str], state: dict, live: dict) -> dict:
    """Refresh stale live snapshot from master.log START lines when process is active."""
    import re
    from datetime import datetime, timezone

    out = dict(live or {})
    lines = recent_or_all or []
    start_line = None
    for line in reversed(lines):
        if " | START " in line:
            start_line = line
            break
        if "THINK_TRAIN_END" in line or "THINK_TRAIN_BEGIN" in line:
            break
    if not start_line:
        return out

    lower = start_line.lower()
    if "navuryx" in lower:
        product = "navuryx"
    elif "navine" in lower:
        product = "navine"
    else:
        product = out.get("product")

    label = start_line.split(" | START ", 1)[-1].strip()
    stage = out.get("stage")
    steps_total = int(out.get("steps_total") or 0)
    m = re.search(r"text:([a-z0-9_\-]+):(\d+)", label, re.I)
    if m:
        stage = m.group(1)
        steps_total = int(m.group(2))
    else:
        m2 = re.search(r"(image_ft|video_ft|voice_calibrate)(?::(\d+))?", label, re.I)
        if m2:
            stage = m2.group(1).replace("_ft", "").replace("_calibrate", "")
            if m2.group(2):
                steps_total = int(m2.group(2))

    cycle_from_state = int(state.get("cycle") or 0)
    cycle_line = None
    for line in reversed(lines):
        if "===== CYCLE " in line:
            cycle_line = line
            break
    if cycle_line:
        cm = re.search(r"CYCLE\s+(\d+)/(\d+)", cycle_line)
        if cm:
            cycle_from_state = max(cycle_from_state, int(cm.group(1)))
            out["max_cycles"] = int(cm.group(2))

    live_updated = out.get("updated_at")
    stale = True
    if live_updated:
        try:
            ts = datetime.fromisoformat(str(live_updated).replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            stale = age > 90
        except Exception:
            stale = True
    if int(out.get("cycle") or 0) < cycle_from_state:
        stale = True

    if stale or not out.get("stage"):
        out["product"] = product
        out["stage"] = stage
        out["label"] = label
        out["steps_total"] = steps_total
        out["cycle"] = cycle_from_state or int(out.get("cycle") or 0)
        out["running"] = True
        # Approximate percent from cycle + stage catalog position
        stage_name = (stage or "").lower()
        try:
            stage_i = STAGE_CATALOG.index(stage_name) if stage_name in STAGE_CATALOG else 0
        except Exception:
            stage_i = 0
        spp = max(1, len(STAGE_CATALOG))
        product_i = 0 if product == "navuryx" else 1
        total = spp * 2
        idx = product_i * spp + stage_i
        out["stage_index"] = idx
        out["stage_total"] = total
        out["percent"] = round(min(99.0, max(0.0, (idx / total) * 100.0)), 1)
        out["updated_at"] = datetime.now(timezone.utc).isoformat()
    return out


def _public_stages() -> list[str]:
    return filter_foreign_brand(STAGE_CATALOG)


def _public_reply(text: str, allow_emoji: bool = True) -> str:
    cleaned = scrub_cross_brand_text(text)
    cleaned = scrub_chat_dashes(cleaned)
    try:
        from navine.text.chat import (
            _is_garbage_output,
            _looks_like_fragment_chain,
            looks_like_encyclopedia_dump,
            looks_like_rag_junk,
        )

        if cleaned and (
            _is_garbage_output(cleaned)
            or _looks_like_fragment_chain(cleaned)
            or looks_like_encyclopedia_dump(cleaned)
            or looks_like_rag_junk(cleaned)
        ):
            if not re.search(r"(?i)\b(?:host pc|motherboard|vram|cpu load)\b", cleaned):
                cleaned = (
                    "I could not form a clear answer from the model just now. "
                    "Ask again with a bit more detail."
                )
    except Exception:
        pass
    if allow_emoji:
        return cleaned
    try:
        from navine.text.chat import strip_emojis

        return strip_emojis(cleaned)
    except Exception:
        return cleaned


def scrub_chat_dashes(text: str) -> str:
    if not text:
        return text
    parts = []
    cursor = 0
    for match in re.finditer(r"```[\w+-]*\n.*?```", text, flags=re.S):
        parts.append(_scrub_dash_chunk(text[cursor : match.start()]))
        parts.append(match.group(0))
        cursor = match.end()
    parts.append(_scrub_dash_chunk(text[cursor:]))
    return "".join(parts)


def _scrub_dash_chunk(chunk: str) -> str:
    out = chunk.replace("\u2014", " - ").replace("\u2013", "-")
    out = re.sub(r"(?<![-])--(?![-])", " - ", out)
    out = re.sub(r" {2,}", " ", out)
    return out



@router.get("/engine")
def engine_status():
    from navine.engine import get_engine

    data = get_engine().status()
    loaded = data.get("loaded") or []
    if isinstance(loaded, list):
        data["loaded"] = filter_foreign_brand(loaded)
    return data


@router.get("/runtime")
def runtime_place():
    from navine.utils.runtime_place import runtime_split_summary

    return runtime_split_summary(client_local=True)


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@router.get("/sandbox")
def sandbox_info():
    from navine.sandbox import sandbox_status

    return sandbox_status()


@router.get("/models/stats", response_model=ModelsStatsResponse)
def models_stats():
    from navine.utils.model_stats import collect_model_stats, load_cached_model_stats

    def _cache_usable(payload: dict | None) -> bool:
        if not payload or not payload.get("models"):
            return False
        names = {str(m.get("name") or "") for m in (payload.get("models") or []) if isinstance(m, dict)}
        if "music" not in names:
            return False
        if not payload.get("capabilities"):
            return False
        if not payload.get("host_specs"):
            return False
        return True

    cached = load_cached_model_stats(max_age_seconds=900)
    if _cache_usable(cached):
        try:
            return ModelsStatsResponse(**cached)
        except Exception:
            pass
    try:
        return ModelsStatsResponse(**collect_model_stats())
    except Exception:
        stale = load_cached_model_stats(max_age_seconds=7 * 24 * 3600)
        if _cache_usable(stale):
            return ModelsStatsResponse(**stale)
        if stale and stale.get("models"):
            return ModelsStatsResponse(**stale)
        raise


@router.get("/info", response_model=InfoResponse)
def info():
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    brand = load_brand()
    model_ids = list(brand.get("model_ids") or [
        "text_enterprise",
        "text_code",
        "image_enterprise",
        "video_enterprise",
        "voice",
    ])
    models = [_model_status(name) for name in model_ids]
    from navine.train.registry import list_training_types

    training_rows = filter_foreign_brand(
        list_training_types(),
        getter=lambda row: f"{row.get('name', '')} {row.get('description', '')}",
    )
    if is_python_only():
        training_rows = [
            row for row in training_rows if not python_train_denied(str(row.get("name") or ""))
        ]
    training_types = [TrainingTypeInfo(**row) for row in training_rows]
    features = brand_ui_features()
    try:
        from navine.utils.runtime_place import runtime_split_summary, runtime_welcome_line

        runtime = runtime_split_summary(client_local=True)
        welcome = runtime_welcome_line()
    except Exception:
        runtime = {
            "ui": "browser",
            "ui_on_visitor_device": True,
            "inference_host": "host",
            "inference_on_visitor_device": False,
            "summary": (
                "The chat UI runs in your browser on your device. "
                "Models run on the Navine host that serves this site."
            ),
        }
        welcome = (
            "UI in your browser | models on the Navine host | "
            "image/video use Navine-trained weights only"
        )
    tagline = welcome
    return InfoResponse(
        pytorch=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        gpu=gpu,
        models=models,
        local_only=is_local_only(),
        unrestricted=True,
        product=str(brand.get("display_name") or "Navine AI - Python"),
        engine=str(brand.get("engine_name") or ""),
        default_text_model=str(brand.get("default_text_model") or "text_enterprise"),
        port=int(brand.get("port") or 8765),
        modes=list(brand.get("modes") or ["chat", "code", "think", "detective", "analyze", "osint", "conscience"]),
        training_types=training_types if features.get("train", True) else [],
        theme_id=str(brand.get("theme_id") or "navine"),
        theme=dict(brand.get("theme") or {}),
        public_url=str(brand.get("public_url") or "").strip() or None,
        tagline=tagline,
        backend=brand_backend(),
        python_only=is_python_only(),
        ui_tabs=brand_ui_tabs(),
        ui_features=features,
        runtime=runtime,
    )


@router.post("/train/admin-access", response_model=AuthLoginResponse)
def train_admin_access(req: AdminUsernameRequest):
    from navine.auth.admin import verify_admin_username
    from navine.auth.users import create_session_token

    name = req.username.strip()
    if not verify_admin_username(name):
        raise HTTPException(status_code=403, detail="Unknown admin username")
    token, expires = create_session_token(name)
    return AuthLoginResponse(token=token, username=name, expires_at=expires, is_admin=True)


@router.post("/auth/signup", response_model=AuthLoginResponse)
def auth_signup(req: AuthSignupRequest):
    from navine.auth.users import UserStore, create_session_token

    store = UserStore()
    try:
        store.signup(req.username, req.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    name = req.username.strip()
    token, expires = create_session_token(name)
    return AuthLoginResponse(token=token, username=name, expires_at=expires, is_admin=False)


@router.post("/auth/login", response_model=AuthLoginResponse)
def auth_login(req: AuthLoginRequest):
    from navine.auth.users import UserStore, create_session_token

    store = UserStore()
    name = req.username.strip()
    if not store.verify(name, req.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token, expires = create_session_token(name)
    return AuthLoginResponse(
        token=token,
        username=name,
        expires_at=expires,
        is_admin=store.is_admin(name),
    )


@router.get("/auth/session", response_model=AuthSessionResponse)
def auth_session(request: Request):
    from navine.auth.users import UserStore

    token = _session_token_from_request(request)
    user = _session_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return AuthSessionResponse(
        username=user,
        expires_at=_token_expires(token),
        is_admin=UserStore().is_admin(user),
    )


@router.get("/downloads")
def downloads_list():
    from navine.downloads import ensure_download_packages

    return {"downloads": ensure_download_packages()}


@router.get("/downloads/{kind}")
def downloads_get(kind: str):
    from navine.downloads import downloads_dir, find_download

    row = find_download(kind)
    if not row:
        raise HTTPException(status_code=404, detail="Download not found")
    path = downloads_dir() / row["filename"]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File missing")
    return FileResponse(
        path,
        filename=row["filename"],
        media_type="application/zip",
    )


@router.get("/user/settings", response_model=UserSettingsResponse)
def user_settings_get(request: Request):
    from navine.user_settings import load_user_settings

    user = _session_user(request) or "guest"
    data = load_user_settings(user)
    return UserSettingsResponse(
        allow_emoji=bool(data.get("allow_emoji", True)),
        discord_webhook_url="",
        discord_webhook_configured=False,
    )


@router.put("/user/settings", response_model=UserSettingsResponse)
def user_settings_put(request: Request, req: UserSettings):
    from navine.user_settings import save_user_settings

    user = _session_user(request) or "guest"
    data = save_user_settings(
        user,
        {
            "allow_emoji": req.allow_emoji,
        },
    )
    return UserSettingsResponse(
        allow_emoji=bool(data.get("allow_emoji", True)),
        discord_webhook_url="",
        discord_webhook_configured=False,
    )


@router.post("/discord/webhook")
def discord_webhook_post(request: Request, payload: dict):
    from navine.integrations.discord_webhook import get_webhook_url, send_webhook, set_webhook_url
    from navine.user_settings import load_user_settings
    from navine.utils.brand import load_brand

    content = str((payload or {}).get("content") or (payload or {}).get("message") or "").strip()
    url = str((payload or {}).get("url") or "").strip() or None
    if not url:
        user = _session_user(request) or "guest"
        url = str(load_user_settings(user).get("discord_webhook_url") or "").strip() or None
    if url:
        set_webhook_url(url)
    if not get_webhook_url(url):
        raise HTTPException(
            status_code=400,
            detail="No Discord webhook configured. Paste the URL in Settings and click Save webhook.",
        )
    brand = load_brand()
    result = send_webhook(content, webhook_url=url, username=str(brand.get("name") or "Navine AI - Python"))
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=str(result.get("error") or "Webhook send failed"))
    return result


@router.get("/mcp/status")
def mcp_status_get():
    from navine.mcp.client import status as mcp_status

    return mcp_status()


@router.get("/mcp/servers")
def mcp_servers_get():
    from navine.mcp.client import list_servers

    return {"servers": list_servers()}


@router.get("/mcp/tools/{server}")
def mcp_tools_get(server: str):
    from navine.mcp.client import list_tools

    result = list_tools(server)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=str(result.get("error") or "MCP tools failed"))
    return result


@router.post("/mcp/call")
def mcp_call_post(payload: dict):
    from navine.mcp.client import call_tool

    server = str((payload or {}).get("server") or "").strip()
    tool = str((payload or {}).get("tool") or "").strip()
    arguments = (payload or {}).get("arguments") or {}
    if not server or not tool:
        raise HTTPException(status_code=400, detail="server and tool are required")
    if not isinstance(arguments, dict):
        raise HTTPException(status_code=400, detail="arguments must be an object")
    result = call_tool(server, tool, arguments)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=str(result.get("error") or "MCP call failed"))
    return result


@router.get("/mcp/cursor-config")
def mcp_cursor_config_get():
    from navine.mcp.server import cursor_mcp_snippet

    return cursor_mcp_snippet()


@router.post("/mcp/servers")
def mcp_servers_upsert(req: McpServerUpsertRequest):
    from navine.mcp.config import upsert_server

    try:
        return upsert_server(
            req.name,
            command=str(req.command or ""),
            args=list(req.args or []),
            url=str(req.url or ""),
            enabled=bool(req.enabled),
            cwd=str(req.cwd or ""),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/mcp/servers/{name}", response_model=McpServerRemoveResponse)
def mcp_servers_delete(name: str):
    from navine.mcp.config import remove_server

    try:
        result = remove_server(name)
        return McpServerRemoveResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/learn/hf/datasets")
def learn_hf_datasets_search(search: str = "", limit: int = 20):
    from navine.learn.hf_search import search_datasets

    q = (search or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="search query is required")
    try:
        rows = search_datasets(q, limit=limit)
        return {"ok": True, "query": q, "count": len(rows), "datasets": rows}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Hugging Face search failed: {exc}") from exc


@router.get("/learn/hf/curated")
def learn_hf_datasets_curated():
    from navine.learn.hf_search import curated_dataset_search

    try:
        return {"ok": True, "groups": curated_dataset_search()}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Hugging Face curated search failed: {exc}") from exc


def _persist_user_chat(
    request: Request,
    session_id: Optional[str],
    user_message: str,
    assistant: str,
    meta: Optional[dict] = None,
) -> Optional[str]:
    username = _session_user(request)
    if not username:
        return session_id
    try:
        from navine.auth.chats import append_exchange, create_chat, get_chat

        chat_id = session_id
        if not chat_id or not get_chat(username, chat_id):
            created = create_chat(username)
            chat_id = created["id"]
        append_exchange(username, chat_id, user_message, assistant, meta=meta)
        return chat_id
    except Exception:
        return session_id


@router.get("/chats", response_model=UserChatListResponse)
def chats_list(request: Request):
    from navine.auth.chats import list_chats

    user = _require_user(request)
    rows = list_chats(user)
    return UserChatListResponse(
        chats=[
            UserChatSummary(
                id=str(row.get("id") or ""),
                title=str(row.get("title") or "New chat"),
                created_at=row.get("created_at"),
                updated_at=row.get("updated_at"),
                message_count=int(row.get("message_count") or 0),
            )
            for row in rows
            if row.get("id")
        ]
    )


@router.post("/chats", response_model=UserChatSummary)
def chats_create(request: Request, req: UserChatCreateRequest):
    from navine.auth.chats import create_chat

    user = _require_user(request)
    row = create_chat(user, title=req.title)
    return UserChatSummary(
        id=row["id"],
        title=row["title"],
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        message_count=int(row.get("message_count") or 0),
    )


@router.get("/chats/{chat_id}", response_model=UserChatDetailResponse)
def chats_get(chat_id: str, request: Request):
    from navine.auth.chats import get_chat

    user = _require_user(request)
    chat = get_chat(user, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    messages = []
    for msg in chat.get("messages") or []:
        messages.append(
            UserChatMessage(
                role=str(msg.get("role") or "user"),
                content=str(msg.get("content") or ""),
                timestamp=msg.get("timestamp"),
            )
        )
    return UserChatDetailResponse(
        id=str(chat.get("id") or chat_id),
        title=str(chat.get("title") or "New chat"),
        created_at=chat.get("created_at"),
        updated_at=chat.get("updated_at"),
        messages=messages,
    )


@router.patch("/chats/{chat_id}", response_model=UserChatSummary)
def chats_rename(chat_id: str, request: Request, req: UserChatRenameRequest):
    from navine.auth.chats import rename_chat

    user = _require_user(request)
    row = rename_chat(user, chat_id, req.title)
    if not row:
        raise HTTPException(status_code=404, detail="Chat not found")
    return UserChatSummary(
        id=row["id"],
        title=row["title"],
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        message_count=int(row.get("message_count") or 0),
    )


@router.delete("/chats/{chat_id}")
def chats_delete(chat_id: str, request: Request):
    from navine.auth.chats import delete_chat

    user = _require_user(request)
    ok = delete_chat(user, chat_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"ok": True}


@router.post("/admin/login", response_model=AdminLoginResponse)
def admin_login(req: AdminLoginRequest):
    from navine.auth.users import UserStore, create_session_token

    store = UserStore()
    name = req.username.strip()
    if not store.verify(name, req.password):
        raise HTTPException(status_code=401, detail="Invalid admin username or password")
    if not store.is_admin(name):
        raise HTTPException(status_code=403, detail="Admin access required")
    token, expires = create_session_token(name)
    return AdminLoginResponse(token=token, username=name, expires_at=expires)


@router.get("/admin/session", response_model=AdminSessionResponse)
def admin_session(request: Request):
    from navine.auth.users import UserStore

    token = _session_token_from_request(request)
    user = _session_user(request)
    if not user or not UserStore().is_admin(user):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return AdminSessionResponse(
        username=user,
        expires_at=_token_expires(token),
        is_admin=True,
    )


@router.get("/train/jobs", response_model=TrainJobsResponse)
def train_jobs(request: Request):
    _require_train_feature()
    _require_admin(request)
    from navine.api.train_jobs import list_recent_jobs

    return TrainJobsResponse(jobs=list_recent_jobs())


@router.post("/train/start", response_model=TrainJobResponse)
def train_start(req: TrainStartRequest, request: Request):
    _require_train_feature()
    admin_user = _require_admin(request)
    from navine.api.train_jobs import spawn_train_job

    media = _resolve_train_media(req)
    if python_train_denied(req.target):
        raise HTTPException(status_code=403, detail="This training target is not available on python-only sites")
    try:
        job = spawn_train_job(
            target=req.target,
            steps=req.steps,
            config=req.config,
            language=req.language,
            require_cuda=bool(req.require_cuda),
            started_by=admin_user,
            voice_name=req.voice_name,
            voice_sample=media.get("voice_sample", req.voice_sample),
            voice_set_default=bool(req.voice_set_default),
            deepfake_strength=req.deepfake_strength,
            deepfake_source=media.get("deepfake_source"),
            deepfake_target=media.get("deepfake_target"),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return TrainJobResponse(job=job)


@router.post("/train/custom", response_model=TrainJobResponse)
def train_custom(req: TrainCustomRequest, request: Request):
    _require_train_feature()
    admin_user = _require_admin(request)
    from navine.api.train_jobs import spawn_train_job

    custom = {
        "source": req.source,
        "url": req.url,
        "text": req.text,
        "title": req.title,
    }
    media = _resolve_train_media(req)
    if python_train_denied(req.train_target):
        raise HTTPException(status_code=403, detail="This training target is not available on python-only sites")
    try:
        job = spawn_train_job(
            target=req.train_target,
            steps=req.steps,
            config=req.config,
            language=req.language,
            require_cuda=bool(req.require_cuda),
            custom=custom,
            started_by=admin_user,
            voice_name=req.voice_name,
            voice_sample=media.get("voice_sample", req.voice_sample),
            voice_set_default=bool(req.voice_set_default),
            deepfake_strength=req.deepfake_strength,
            deepfake_source=media.get("deepfake_source"),
            deepfake_target=media.get("deepfake_target"),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TrainJobResponse(job=job)


@router.post("/train/learn", response_model=TrainJobResponse)
def train_learn(req: TrainLearnRequest, request: Request):
    _require_train_feature()
    admin_user = _require_admin(request)
    from navine.api.train_jobs import spawn_learn_job

    try:
        job = spawn_learn_job(
            action=req.action,
            url=req.url,
            text=req.text,
            title=req.title,
            max_pages=req.max_pages,
            started_by=admin_user,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TrainJobResponse(job=job)


@router.get("/train/progress", response_model=TrainProgressResponse)
def train_progress():
    from navine.utils.paths import get_project_root
    from navine.utils.training_lock import training_lock_active

    root = get_project_root()
    log_dir = root / "logs" / "think_train"
    shared = root.parent / "Navine AI - Python" / "logs" / "think_train"
    if (shared / "progress.live.json").exists() or (shared / "state.json").exists() or (shared / "master.log").exists():
        if root.name != "Navine AI - Python" and shared.exists():
            log_dir_candidates = [shared, log_dir]
        else:
            log_dir_candidates = [log_dir, shared]
    else:
        log_dir_candidates = [log_dir]

    state: dict = {}
    live: dict = {}
    recent: list = []
    all_lines: list = []
    last_line = None
    for candidate in log_dir_candidates:
        state_path = candidate / "state.json"
        live_path = candidate / "progress.live.json"
        master = candidate / "master.log"
        if state_path.exists() or live_path.exists() or master.exists():
            if state_path.exists():
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except Exception:
                    state = {}
            if live_path.exists():
                try:
                    live = json.loads(live_path.read_text(encoding="utf-8"))
                except Exception:
                    live = {}
            if master.exists():
                try:
                    all_lines = master.read_text(encoding="utf-8", errors="ignore").splitlines()
                    recent = all_lines[-8:] if all_lines else []
                    last_line = all_lines[-1] if all_lines else None
                except Exception:
                    pass
            break

    locks = {
        name: bool(training_lock_active(name))
        for name in ("text", "image", "video", "voice", "deepfake")
    }
    running = bool(live.get("running"))
    if not running and any(locks.values()):
        running = True
    if not running:
        try:
            import psutil

            for proc in psutil.process_iter(["cmdline"]):
                cmd = " ".join(proc.info.get("cmdline") or [])
                if "run_think_train_loop" in cmd or "start_think_train" in cmd:
                    running = True
                    break
        except Exception:
            if last_line and "THINK_TRAIN_END" not in (last_line or "") and (
                "START" in (last_line or "") or "CYCLE" in (last_line or "")
            ):
                running = True
    if last_line and "THINK_TRAIN_END" in last_line:
        running = False
        live["running"] = False

    if running and all_lines:
        live = _parse_master_live(all_lines[-80:], state, live)

    from navine.api.train_progress_helper import enrich_progress_from_admin_jobs

    live = enrich_progress_from_admin_jobs(live, locks)
    if live.get("running"):
        running = True

    models = [_model_status(name) for name in ("text_enterprise", "text_code", "image_enterprise", "video_enterprise", "voice")]

    products_live = live.get("products") or {}
    best = dict(state.get("best_scores") or {})
    product_list = []
    brand = load_brand()
    theme_key = str(brand.get("theme_id") or brand.get("package") or "navine").lower()
    if "navuryx" in theme_key:
        product_names = ("navuryx",)
    elif "hitboy" in theme_key:
        product_names = ("hitboy",)
    else:
        product_names = ("navine",)
    for pname in product_names:
        meta = products_live.get(pname) or {}
        hist = (state.get("history") or [])
        last_hist = {}
        if hist:
            last_hist = ((hist[-1] or {}).get("products") or {}).get(pname) or {}
        product_list.append(
            ProductProgress(
                name=pname,
                phase=meta.get("phase") or last_hist.get("phase"),
                score=meta.get("score")
                if meta.get("score") is not None
                else (last_hist.get("score") if last_hist.get("score") is not None else best.get(pname)),
                streak=meta.get("streak")
                if meta.get("streak") is not None
                else (state.get("consecutive_pass") or {}).get(pname),
                sample=meta.get("sample") or last_hist.get("sample"),
                active=bool(live.get("product") == pname and running),
            )
        )

    stage = live.get("stage")
    label = live.get("label")
    if is_foreign_brand_text(stage):
        stage = "training"
    if is_foreign_brand_text(label):
        label = "training"
    recent_public = filter_foreign_brand(recent)
    admin_recent = live.get("recent_lines") or []
    if admin_recent:
        recent_public = filter_foreign_brand(admin_recent) or recent_public
    last_public = live.get("last_line") or (last_line if last_line and not is_foreign_brand_text(last_line) else (recent_public[-1] if recent_public else None))
    scores = dict(state.get("best_scores") or {})
    scores = {k: v for k, v in scores.items() if not is_foreign_brand_text(k)}

    marathon_round = int(live.get("marathon_round") or live.get("cycle") or 0) or None
    marathon_loop = bool(live.get("marathon_loop"))
    max_cycles = int(live.get("max_cycles") or 0)
    if marathon_loop and max_cycles <= 0:
        max_cycles = 0

    return TrainProgressResponse(
        running=running,
        cycle=int(live.get("cycle") or state.get("cycle") or 0),
        max_cycles=max_cycles if max_cycles > 0 else None,
        best_scores=scores,
        consecutive_pass={
            k: v
            for k, v in dict(state.get("consecutive_pass") or {}).items()
            if not is_foreign_brand_text(k)
        },
        last_line=last_public,
        recent_lines=recent_public,
        models=models,
        stages=_public_stages(),
        product=str(brand_name()),
        active_product=live.get("product") if str(live.get("product") or "") in product_names else product_names[0],
        stage=stage,
        stage_index=int(live.get("stage_index") or 0),
        stage_total=int(live.get("stage_total") or 0),
        steps=int(live.get("steps") or 0),
        steps_total=int(live.get("steps_total") or 0),
        label=label,
        percent=float(live.get("percent") or 0.0),
        elapsed_s=live.get("elapsed_s"),
        eta_s=live.get("eta_s"),
        updated_at=live.get("updated_at"),
        products=product_list,
        locks=locks,
        marathon_round=marathon_round,
        marathon_loop=marathon_loop,
    )


@router.get("/train/types", response_model=TrainingTypesResponse)
def train_types():
    from navine.train.registry import list_training_types

    rows = filter_foreign_brand(
        list_training_types(),
        getter=lambda row: f"{row.get('name', '')} {row.get('description', '')}",
    )
    if is_python_only():
        rows = [row for row in rows if not python_train_denied(str(row.get("name") or ""))]
    return TrainingTypesResponse(
        product=brand_name(),
        types=[TrainingTypeInfo(**row) for row in rows],
    )


@router.post("/train/comprehensive")
def train_comprehensive(request: Request, body: dict | None = None):
    import subprocess
    import sys

    from navine.utils.brand import brand_name
    from navine.utils.paths import get_project_root

    _require_train_feature()
    _require_admin(request)

    root = get_project_root()
    script = root / "scripts" / "train_comprehensive.py"
    if not script.exists():
        raise HTTPException(status_code=404, detail="train_comprehensive.py not found")
    py = root / "venv" / "Scripts" / "python.exe"
    python = str(py) if py.exists() else sys.executable
    mode = "full"
    if isinstance(body, dict) and body.get("mode"):
        mode = str(body.get("mode"))
    args = [python, str(script), "--mode", mode]
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_CONSOLE
    subprocess.Popen(args, cwd=str(root), creationflags=creationflags)
    return {"ok": True, "started": True, "product": brand_name(), "mode": mode}


@router.get("/autolearn/status", response_model=AutolearnStatusResponse)
def autolearn_status():
    from navine.autolearn.engine import AutolearnEngine

    data = AutolearnEngine.get_status()
    return AutolearnStatusResponse(**data)


@router.get("/marathon/status", response_model=MarathonStatusResponse)
def marathon_status():
    from navine.autolearn.marathon import get_marathon_status

    data = get_marathon_status()
    return MarathonStatusResponse(**data)


@router.get("/host/specs")
def host_specs():
    try:
        from navine.realtime.host_specs import collect_host_specs, format_host_specs_reply

        specs = collect_host_specs()
        return {"ok": True, "specs": specs, "text": format_host_specs_reply(specs)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/chat", response_model=ChatResponse)
def chat_generate(req: ChatRequest, request: Request):
    try:
        from navine.text.chat import chat_with_meta, detect_code_intent, resolve_chat_profile
        from navine.memory.conversations import record_chat_exchange
        from navine.desktop.intents import handle_desktop_chat_message
        from navine.utils.gpu_session import inference_session

        user_message, history = normalize_chat_request(req)
        if not user_message and not getattr(req, "image_base64", None):
            raise HTTPException(status_code=400, detail="No user message found")
        user_message = _apply_chat_image_attachment(user_message, req)
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        try:
            from navine.text.typo_correct import normalize_user_text

            user_message = normalize_user_text(user_message)
        except Exception:
            pass
        desktop = handle_desktop_chat_message(user_message)
        if desktop is not None:
            if not desktop.get("ok"):
                raise HTTPException(status_code=403, detail=str(desktop.get("error") or "Desktop action failed"))
            result = _public_reply(str(desktop.get("text") or desktop.get("summary") or "Done."))
            session_id = record_chat_exchange(
                user_message,
                result,
                session_id=req.session_id,
                meta={"desktop_action": desktop.get("action"), "path": desktop.get("path")},
            )
            session_id = _persist_user_chat(
                request,
                session_id or req.session_id,
                user_message,
                result,
                meta={"desktop_action": desktop.get("action")},
            )
            return ChatResponse(
                text=result,
                message=ChatMessage(role="assistant", content=result),
                is_code=False,
                searched=False,
                search_attempted=False,
                session_id=session_id or req.session_id,
            )
        use_search = req.use_search
        if not use_search:
            try:
                from navine.memory.config import get_search_config
                from navine.search.web import detect_search_intent, is_explicit_search_request

                cfg = get_search_config()
                use_search = bool(
                    cfg.get("default_enabled")
                    or is_explicit_search_request(user_message)
                    or (
                        cfg.get("auto_search_factual", False)
                        and detect_search_intent(user_message) == "factual"
                    )
                )
            except Exception:
                use_search = False
        profile = resolve_chat_profile(user_message, req.model_profile)
        with inference_session("chat"):
            result, meta = chat_with_meta(
                user_message,
                history=history,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                use_rag=req.use_rag,
                use_search=use_search,
                model_profile=req.model_profile,
                session_id=req.session_id,
                allow_emoji=bool(req.allow_emoji),
            )
        result = _public_reply(result or "", allow_emoji=bool(req.allow_emoji))
        if not str(result).strip():
            result = (
                "I am here, but the model returned an empty reply. "
                "Try again in a moment, or pause training if the GPU is busy."
            )
        session_id = record_chat_exchange(
            user_message,
            result,
            session_id=req.session_id,
            meta=meta,
        )
        session_id = _persist_user_chat(request, session_id or req.session_id, user_message, result, meta=meta)
        return ChatResponse(
            text=result,
            message=ChatMessage(role="assistant", content=result),
            is_code=detect_code_intent(user_message) or meta.get("model_profile") == "code",
            searched=meta.get("searched", False),
            search_attempted=meta.get("search_attempted", False),
            session_id=session_id or req.session_id,
            model_profile=meta.get("model_profile"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/chat/stream")
def chat_stream(req: ChatRequest, request: Request):
    try:
        from navine.text.chat import chat_with_meta, detect_code_intent, resolve_chat_profile
        from navine.memory.conversations import record_chat_exchange
        from navine.desktop.intents import handle_desktop_chat_message

        user_message, history = normalize_chat_request(req)
        if not user_message and not getattr(req, "image_base64", None):
            raise HTTPException(status_code=400, detail="No user message found")
        user_message = _apply_chat_image_attachment(user_message, req)
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        try:
            from navine.text.typo_correct import normalize_user_text

            user_message = normalize_user_text(user_message)
        except Exception:
            pass

        def event_gen():
            from navine.utils.gpu_session import inference_session

            profile = resolve_chat_profile(user_message, req.model_profile)
            yield f"data: {json.dumps({'type': 'meta', 'model_profile': profile})}\n\n"
            try:
                desktop = handle_desktop_chat_message(user_message)
                if desktop is not None:
                    if not desktop.get("ok"):
                        yield f"data: {json.dumps({'type': 'error', 'detail': str(desktop.get('error') or 'Desktop action failed')})}\n\n"
                        return
                    result = _public_reply(str(desktop.get("text") or desktop.get("summary") or "Done."))
                    if not str(result).strip():
                        result = "Done."
                    yield f"data: {json.dumps({'type': 'token', 'text': result})}\n\n"
                    session_id = record_chat_exchange(
                        user_message,
                        result,
                        session_id=req.session_id,
                        meta={"desktop_action": desktop.get("action"), "path": desktop.get("path")},
                    )
                    session_id = _persist_user_chat(
                        request,
                        session_id or req.session_id,
                        user_message,
                        result,
                        meta={"desktop_action": desktop.get("action")},
                    )
                    yield f"data: {json.dumps({'type': 'done', 'text': result, 'session_id': session_id or req.session_id, 'model_profile': profile, 'is_code': False})}\n\n"
                    return
                use_search = req.use_search
                if not use_search:
                    try:
                        from navine.memory.config import get_search_config
                        from navine.search.web import is_explicit_search_request

                        cfg = get_search_config()
                        use_search = bool(cfg.get("default_enabled") or is_explicit_search_request(user_message))
                    except Exception:
                        use_search = False
                with inference_session("chat"):
                    result, meta = chat_with_meta(
                        user_message,
                        history=history,
                        max_new_tokens=req.max_tokens,
                        temperature=req.temperature,
                        use_rag=req.use_rag,
                        use_search=use_search,
                        model_profile=req.model_profile,
                        session_id=req.session_id,
                        allow_emoji=bool(req.allow_emoji),
                    )
                result = _public_reply(result or "", allow_emoji=bool(req.allow_emoji))
                if not str(result).strip():
                    result = (
                        "I am here, but the model returned an empty reply. "
                        "Try again in a moment, or pause training if the GPU is busy."
                    )
                chunk = 6
                for i in range(0, len(result), chunk):
                    yield f"data: {json.dumps({'type': 'token', 'text': result[i:i + chunk]})}\n\n"
                session_id = record_chat_exchange(
                    user_message,
                    result,
                    session_id=req.session_id,
                    meta=meta,
                )
                session_id = _persist_user_chat(request, session_id or req.session_id, user_message, result, meta=meta)
                yield f"data: {json.dumps({'type': 'done', 'text': result, 'session_id': session_id or req.session_id, 'model_profile': meta.get('model_profile') or profile, 'is_code': detect_code_intent(user_message) or meta.get('model_profile') == 'code', 'searched': meta.get('searched', False)})}\n\n"
            except FileNotFoundError as exc:
                yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"
            except Exception as exc:
                detail = str(exc) or type(exc).__name__
                lower = detail.lower()
                if "out of memory" in lower or "cuda" in lower:
                    detail = "GPU is busy (training or VRAM). Pause training or retry in a moment."
                yield f"data: {json.dumps({'type': 'error', 'detail': detail})}\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/desktop/status", response_model=DesktopStatusResponse)
def desktop_status():
    from navine.desktop.config import get_desktop_status

    return DesktopStatusResponse(**get_desktop_status())


@router.post("/desktop/settings", response_model=DesktopStatusResponse)
def desktop_settings(req: DesktopSettingsRequest):
    from navine.desktop.config import get_desktop_status, set_desktop_flags

    if req.allow_screen is None and req.allow_input is None:
        raise HTTPException(status_code=400, detail="Provide allow_screen and/or allow_input")
    set_desktop_flags(allow_screen=req.allow_screen, allow_input=req.allow_input)
    return DesktopStatusResponse(**get_desktop_status())


@router.post("/screen/capture")
def screen_capture(req: ScreenCaptureRequest):
    from navine.desktop.capture import capture_screen

    result = capture_screen(path=req.path)
    if not result.get("ok"):
        raise HTTPException(status_code=403, detail=str(result.get("error") or "Capture failed"))
    payload = dict(result)
    try:
        path = Path(str(result.get("path") or ""))
        if path.exists():
            data = path.read_bytes()
            payload["mime"] = "image/png"
            payload["data_base64"] = base64.b64encode(data).decode("ascii")
    except Exception:
        pass
    return payload


@router.post("/screen/describe")
def screen_describe(req: ScreenDescribeRequest):
    from navine.desktop.vision import describe_screen

    result = describe_screen(question=req.question, use_model=req.use_model)
    if not result.get("ok"):
        raise HTTPException(status_code=403, detail=str(result.get("error") or "Describe failed"))
    return result


@router.post("/desktop/input")
def desktop_input(req: DesktopInputRequest):
    from navine.desktop.input_control import click_at, press_hotkey, type_text

    action = req.action
    if action == "type":
        result = type_text(req.text or "")
    elif action == "click":
        if req.x is None or req.y is None:
            raise HTTPException(status_code=400, detail="click requires x and y")
        result = click_at(req.x, req.y, button=req.button, clicks=req.clicks)
    elif action == "hotkey":
        result = press_hotkey(req.keys or "")
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
    if not result.get("ok"):
        raise HTTPException(status_code=403, detail=str(result.get("error") or "Input blocked"))
    return result


@router.post("/text", response_model=TextResponse)
def text_generate(req: TextRequest):
    try:
        from navine.text.chat import chat
        from navine.utils.gpu_session import inference_session

        with inference_session("text"):
            result = _public_reply(chat(
                req.prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
            ))
        return TextResponse(text=result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _clear_media_profile(profile: Optional[str]) -> Optional[str]:
    return None


@router.post("/image", response_model=ImageResponse)
def image_generate(req: ImageRequest):
    try:
        from navine.image.infer import generate
        from navine.image.prompts import enhance_prompt
        from navine.utils.gpu_session import inference_session

        num_steps = req.resolved_steps() if hasattr(req, "resolved_steps") else req.num_steps
        if num_steps is None:
            num_steps = 20
        else:
            num_steps = max(8, min(int(num_steps), 48))
        media_profile = _clear_media_profile(req.model_profile)
        prompt = req.prompt
        try:
            from navine.text.typo_correct import normalize_user_text

            prompt = normalize_user_text(prompt)
        except Exception:
            prompt = req.prompt
        with inference_session("image"):
            path = generate(
                prompt,
                num_steps=num_steps,
                guidance_scale=req.guidance_scale,
                seed=req.seed,
                enhance=req.enhance,
                config_name=None,
                model_profile=media_profile,
            )
        data = Path(path).read_bytes()
        enhanced = enhance_prompt(prompt, profile=media_profile) if req.enhance else prompt
        meta_path = Path(path).with_suffix(".json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        return ImageResponse(
            path=str(path),
            mime="image/png",
            data_base64=base64.b64encode(data).decode("ascii"),
            enhanced_prompt=enhanced,
            reference_used=bool(meta.get("reference_used")),
            keyword_matched=meta.get("keyword_matched"),
            reference_category=meta.get("reference_category"),
            reference_strength=meta.get("reference_strength"),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/video", response_model=VideoResponse)
def video_generate(req: VideoRequest):
    try:
        from navine.video.infer import generate
        from navine.utils.gpu_session import inference_session

        num_frames = req.num_frames
        if num_frames is not None:
            num_frames = max(4, min(int(num_frames), 24))
        media_profile = _clear_media_profile(req.model_profile)
        with inference_session("video"):
            path = generate(
                req.prompt,
                num_frames=num_frames,
                fps=req.fps,
                seed=req.seed,
                config_name=None,
                with_audio=bool(req.audio),
                narration=req.narration,
                voice=req.voice,
                model_profile=media_profile,
            )
        data = Path(path).read_bytes()
        suffix = Path(path).suffix.lower()
        mime = "video/mp4"
        if suffix == ".webm":
            mime = "video/webm"
        elif suffix == ".gif":
            mime = "image/gif"
        meta_path = Path(path).with_suffix(".json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        return VideoResponse(
            path=str(path),
            mime=mime,
            data_base64=base64.b64encode(data).decode("ascii"),
            reference_used=bool(meta.get("reference_used")),
            keyword_matched=meta.get("keyword_matched"),
            reference_category=meta.get("reference_category"),
            reference_strength=meta.get("reference_strength"),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/code", response_model=CodeResponse)
def code_generate(req: CodeRequest):
    try:
        from navine.code.generate import generate_code

        code, lang = generate_code(req.task, language=req.language)
        return CodeResponse(code=code, language=lang)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/code/run", response_model=CodeRunResponse)
def code_run(req: CodeRunRequest):
    try:
        from navine.code.run import run_code

        result = run_code(req.code, req.language, timeout=req.timeout or 20.0)
        return CodeRunResponse(
            ok=bool(result.get("ok")),
            stdout=str(result.get("stdout") or ""),
            stderr=str(result.get("stderr") or ""),
            exit_code=int(result.get("exit_code") or 0),
            language=str(result.get("language") or req.language),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/deepfake/face")
def deepfake_face(req: DeepfakeFaceRequest):
    try:
        from navine.deepfake import swap_face_files
        from navine.utils.media_upload import resolve_image_input

        source_path = resolve_image_input(req.source, req.source_base64, label="source")
        target_path = resolve_image_input(req.target, req.target_base64, label="target")
        path = swap_face_files(str(source_path), str(target_path), strength=req.strength)
        data = Path(path).read_bytes()
        return {
            "path": str(path),
            "mime": "image/png",
            "data_base64": base64.b64encode(data).decode("ascii"),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/deepfake/video")
def deepfake_video_route(req: DeepfakeVideoRequest):
    try:
        from navine.deepfake import deepfake_video

        path = deepfake_video(
            req.source,
            target_video=req.target_video,
            prompt=req.prompt,
            num_frames=req.num_frames,
            fps=req.fps,
            strength=req.strength,
            seed=req.seed,
            with_audio=bool(req.audio),
            narration=req.narration,
        )
        data = Path(path).read_bytes()
        return {
            "path": str(path),
            "mime": "video/mp4",
            "data_base64": base64.b64encode(data).decode("ascii"),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/osint/investigate", response_model=OsintInvestigateResponse)
def osint_investigate(req: OsintInvestigateRequest):
    try:
        from navine.osint import investigate

        result = investigate(
            req.target,
            kind=req.kind,
            use_search=bool(req.use_search),
            output_dir=req.output_dir,
        )
        return OsintInvestigateResponse(
            target=str(result.get("target") or req.target),
            kind=str(result.get("kind") or req.kind),
            query=str(result.get("query") or ""),
            analysis=str(result.get("analysis") or ""),
            search_results=list(result.get("search_results") or []),
            username_scan=result.get("username_scan"),
            domain_intel=result.get("domain_intel"),
            ip_intel=result.get("ip_intel"),
            report_path=result.get("report_path"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/voice/list")
def voice_list():
    try:
        from navine.assistant.voice import list_voices

        return {"voices": list_voices()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/voice/status")
def voice_status_route():
    try:
        from navine.assistant.voice import voice_status

        return voice_status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/voice/clone")
def voice_clone(req: VoiceCloneRequest):
    try:
        from navine.assistant.voice import clone_voice

        result = clone_voice(
            req.name,
            sample_path=req.sample_path or "",
            language=req.language,
            sample_base64=req.sample_base64,
        )
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=result.get("error") or "clone failed")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/voice/speak")
def voice_speak(req: VoiceSpeakRequest):
    try:
        from navine.assistant.voice import speak

        voice_name = (req.voice or "").strip() or None
        result = speak(req.text, voice=voice_name, play=bool(req.play))
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=result.get("error") or "speak failed")
        payload = {
            "ok": True,
            "path": result.get("path"),
            "voice": result.get("voice"),
            "backend": result.get("backend"),
            "played": result.get("played"),
        }
        path = result.get("path")
        if path and Path(path).exists():
            data = Path(path).read_bytes()
            payload["mime"] = "audio/wav"
            payload["data_base64"] = base64.b64encode(data).decode("ascii")
            payload["bytes"] = len(data)
        return payload
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/music", response_model=MusicResponse)
def music_generate(req: MusicRequest):
    try:
        from navine.music.generate import generate_music

        result = generate_music(req.prompt, duration=float(req.duration), seed=req.seed)
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=result.get("error") or "music generation failed")
        path = Path(str(result.get("path") or ""))
        data_b64 = None
        nbytes = int(result.get("bytes") or 0)
        if path.exists():
            raw = path.read_bytes()
            data_b64 = base64.b64encode(raw).decode("ascii")
            nbytes = len(raw)
        return MusicResponse(
            ok=True,
            path=str(path),
            mime=str(result.get("mime") or "audio/wav"),
            data_base64=data_b64,
            bytes=nbytes,
            meta=dict(result.get("meta") or {}),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/voice/transcribe")
def voice_transcribe(req: VoiceTranscribeRequest):
    try:
        from datetime import datetime, timezone

        from navine.assistant.voice import output_dir, transcribe_file

        raw = base64.b64decode(req.audio_base64)
        if not raw:
            raise HTTPException(status_code=400, detail="Empty audio payload")
        mime = (req.mime or "audio/wav").lower()
        ext = ".wav"
        if "webm" in mime:
            ext = ".webm"
        elif "ogg" in mime:
            ext = ".ogg"
        elif "mpeg" in mime or "mp3" in mime:
            ext = ".mp3"
        elif "mp4" in mime or "m4a" in mime:
            ext = ".m4a"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        name = (req.filename or f"upload_{stamp}").strip()
        if not Path(name).suffix:
            name = f"{name}{ext}"
        path = output_dir() / Path(name).name
        path.write_bytes(raw)
        wav_path = path
        if path.suffix.lower() != ".wav":
            try:
                from pydub import AudioSegment

                converted = path.with_suffix(".wav")
                AudioSegment.from_file(path).export(converted, format="wav")
                wav_path = converted
            except Exception:
                wav_path = path
        result = transcribe_file(str(wav_path))
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=result.get("error") or "transcribe failed")
        return {
            "ok": True,
            "text": result.get("text") or "",
            "path": str(wav_path),
            "language": result.get("language"),
            "backend": result.get("backend"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/camera/describe")
def camera_describe(req: CameraDescribeRequest):
    try:
        from navine.desktop.vision import describe_image_bytes, describe_screen

        raw = base64.b64decode(req.image_base64)
        if not raw:
            raise HTTPException(status_code=400, detail="Empty image payload")
        camera = describe_image_bytes(
            raw,
            question=req.question,
            use_model=bool(req.use_model),
            mime=req.mime or "image/jpeg",
            label="camera",
        )
        if not camera.get("ok"):
            raise HTTPException(status_code=400, detail=camera.get("error") or "camera describe failed")
        payload = {
            "ok": True,
            "text": camera.get("text") or camera.get("summary") or "",
            "summary": camera.get("summary"),
            "path": camera.get("path"),
            "analysis": camera.get("analysis"),
            "model_used": camera.get("model_used"),
        }
        if req.include_screen:
            try:
                screen = describe_screen(question=req.question, use_model=bool(req.use_model))
                if screen.get("ok"):
                    payload["screen"] = {
                        "text": screen.get("text"),
                        "path": screen.get("path"),
                        "summary": screen.get("summary"),
                    }
                    if screen.get("text"):
                        payload["text"] = (
                            f"{payload['text']}\n\n--- Screen ---\n{screen.get('text')}"
                        ).strip()
            except Exception as exc:
                payload["screen_error"] = str(exc)
        return payload
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/media/upload")
async def media_upload(file: UploadFile = File(...)):
    try:
        from datetime import datetime, timezone

        from navine.utils.paths import get_project_root

        raw_name = (file.filename or "upload.bin").replace("\\", "/").split("/")[-1]
        safe = "".join(ch for ch in raw_name if ch.isalnum() or ch in "._- ")[:120].strip() or "upload.bin"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = get_project_root() / "outputs" / "uploads"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{stamp}_{safe}"
        data = await file.read()
        if len(data) > 40 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 40MB)")
        path.write_bytes(data)
        mime = file.content_type or "application/octet-stream"
        payload = {
            "ok": True,
            "path": str(path),
            "filename": safe,
            "mime": mime,
            "size": len(data),
            "download_url": f"/api/files/download?path={path.as_posix()}",
        }
        if mime.startswith("image/"):
            payload["data_base64"] = base64.b64encode(data).decode("ascii")
        return payload
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/files/create", response_model=FileCreateResponse)
def files_create(req: FileCreateRequest):
    try:
        from datetime import datetime, timezone
        from urllib.parse import quote

        from navine.utils.paths import get_project_root

        name = req.filename.replace("\\", "/").split("/")[-1]
        safe = "".join(ch for ch in name if ch.isalnum() or ch in "._- ")[:160].strip()
        if not safe:
            raise HTTPException(status_code=400, detail="Invalid filename")
        if "." not in safe:
            safe = f"{safe}.txt"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = get_project_root() / "outputs" / "files"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{stamp}_{safe}"
        raw = req.content.encode("utf-8")
        path.write_bytes(raw)
        mime = req.mime or "text/plain"
        return FileCreateResponse(
            path=str(path),
            filename=safe,
            mime=mime,
            size=len(raw),
            download_url=f"/api/files/download?path={quote(path.as_posix())}",
            data_base64=base64.b64encode(raw).decode("ascii") if len(raw) < 2_000_000 else None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/files/zip", response_model=FileCreateResponse)
def files_zip(req: ZipCreateRequest):
    try:
        from navine.files.zip_util import create_zip_archive, zip_response_payload

        files = [{"path": row.path, "content": row.content} for row in req.files]
        path, filename = create_zip_archive(files, req.filename)
        payload = zip_response_payload(path, filename)
        return FileCreateResponse(**payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/files/download")
def files_download(path: str):
    try:
        from navine.utils.paths import get_project_root

        root = get_project_root().resolve()
        target = Path(path).resolve()
        allowed = [
            (root / "outputs").resolve(),
            (root / "data").resolve(),
        ]
        if not any(str(target).startswith(str(base)) for base in allowed):
            raise HTTPException(status_code=403, detail="Path not allowed")
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(path=str(target), filename=target.name)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/keys", response_model=ApiKeyCreateResponse)
def create_api_key(req: ApiKeyCreateRequest, request: Request):
    try:
        from navine.api.settings import get_key_store

        owner = _require_user(request)
        store = get_key_store()
        key_id, raw_key, entry = store.create(
            name=req.name,
            scopes=req.scopes,
            expires_days=req.expires_days,
            owner=owner,
        )
        return ApiKeyCreateResponse(
            id=key_id,
            name=req.name,
            key=raw_key,
            prefix=entry.get("prefix"),
            scopes=list(entry.get("scopes") or []),
            expires_at=entry.get("expires_at"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/keys", response_model=ApiKeyListResponse)
def list_api_keys(request: Request):
    try:
        from navine.api.settings import get_key_store
        from navine.auth.users import UserStore

        user = _require_user(request)
        store = get_key_store()
        is_admin = UserStore().is_admin(user)
        rows = store.list_keys(owner=None if is_admin else user)
        return ApiKeyListResponse(keys=[ApiKeyInfo(**row) for row in rows])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/keys/{key_id}")
def delete_api_key(key_id: str, request: Request):
    try:
        from navine.api.settings import get_key_store
        from navine.auth.users import UserStore

        user = _require_user(request)
        store = get_key_store()
        is_admin = UserStore().is_admin(user)
        deleted = store.delete(key_id, owner=None if is_admin else user, admin=is_admin)
        if not deleted:
            raise HTTPException(status_code=404, detail="API key not found")
        return {"ok": True, "id": key_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
