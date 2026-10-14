import time
import uuid

from fastapi import APIRouter, HTTPException

from navine.api.chat_utils import run_api_chat
from navine.api.schemas import (
    ChatMessage,
    OpenAIChatChoice,
    OpenAIChatMessage,
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIUsage,
)

openai_router = APIRouter()


@openai_router.get("/models")
def openai_list_models():
    from navine.llm import list_llms

    models = []
    for row in list_llms():
        if not row.get("runnable"):
            continue
        models.append(
            {
                "id": row["id"],
                "object": "model",
                "created": 0,
                "owned_by": "navine-ai",
                "modality": row.get("modality"),
            }
        )
    return {"object": "list", "data": models}


@openai_router.post("/chat/completions", response_model=OpenAIChatResponse)
def openai_chat_completions(req: OpenAIChatRequest):
    if req.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported")
    chat_messages = []
    for item in req.messages:
        role = item.role.lower()
        if role not in ("user", "assistant", "system"):
            role = "user"
        if role == "system":
            chat_messages.append(ChatMessage(role="user", content=f"System: {item.content}"))
        else:
            chat_messages.append(ChatMessage(role=role, content=item.content))
    try:
        model_name = (req.model or "text_enterprise").strip()
        text = run_api_chat(
            chat_messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            model=model_name,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return OpenAIChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        object="chat.completion",
        created=int(time.time()),
        model=model_name,
        choices=[
            OpenAIChatChoice(
                index=0,
                message=OpenAIChatMessage(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=OpenAIUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )
