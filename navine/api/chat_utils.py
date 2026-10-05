from typing import Dict, List, Tuple

from fastapi import HTTPException

from navine.api.schemas import ChatMessage, ChatRequest


def normalize_chat_request(req: ChatRequest) -> Tuple[str, List[Dict[str, str]]]:
    history = [{"user": turn.user, "assistant": turn.assistant} for turn in req.history]
    if req.messages:
        user_message = None
        index = 0
        while index < len(req.messages):
            message = req.messages[index]
            if message.role == "user":
                user_content = message.content
                if index + 1 < len(req.messages) and req.messages[index + 1].role == "assistant":
                    history.append(
                        {
                            "user": user_content,
                            "assistant": req.messages[index + 1].content,
                        }
                    )
                    index += 2
                    continue
                user_message = user_content
            index += 1
        if user_message:
            return user_message, history
        raise ValueError("No user message found in messages")
    if req.message:
        return req.message.strip(), history
    raise ValueError("Provide message or messages")


def run_api_chat(
    messages: list[ChatMessage],
    max_tokens=None,
    temperature=None,
    model: str | None = None,
) -> str:
    from navine.text.chat import chat

    if not messages:
        raise HTTPException(status_code=400, detail="At least one message is required")
    if messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")
    history: List[Dict[str, str]] = []
    index = 0
    while index < len(messages) - 1:
        message = messages[index]
        if message.role == "user":
            user_content = message.content
            if index + 1 < len(messages) - 1 and messages[index + 1].role == "assistant":
                history.append(
                    {
                        "user": user_content,
                        "assistant": messages[index + 1].content,
                    }
                )
                index += 2
                continue
        index += 1
    user_message = messages[-1].content
    use_search = False
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
    key = str(model or "").strip().lower().replace("-", "_")
    profile = None
    if key in ("think", "thinking", "reason", "reasoning"):
        profile = "think"
    elif key in ("detective", "mystery", "investigate", "cipher", "cicada", "arg", "puzzle"):
        profile = "detective"
    elif key in ("analyze", "analyse", "analysis", "breakdown", "inspect"):
        profile = "analyze"
    elif key in ("code", "coding"):
        profile = "code"
    if key in ("hitboyx23_ai", "hitboyx23", "hitboy", "hitboyx23_ai_python", "hitboyx23_python", "hitboy_python"):
        return chat(
            user_message,
            history=history,
            max_new_tokens=max_tokens,
            temperature=temperature,
            use_search=use_search,
            model_profile=profile,
        )
    if key in ("text_code", "code", "coding", "navine_code") or profile == "code":
        from navine.text.infer import generate
        from navine.utils.paths import get_checkpoint_dir

        ckpt = get_checkpoint_dir("text_code") / "latest.pt"
        return generate(
            user_message,
            max_new_tokens=max_tokens,
            temperature=temperature,
            checkpoint=str(ckpt) if ckpt.exists() else None,
            mode="code",
        )
    return chat(
        user_message,
        history=history,
        max_new_tokens=max_tokens,
        temperature=temperature,
        use_search=use_search,
        model_profile=profile,
    )
