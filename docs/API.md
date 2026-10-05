# Navine AI API Reference

Navine AI provides a local REST API for text, chat, code, image, and video generation. All inference runs on your machine with PyTorch. No external AI APIs are used.

## Base URL

Default: `http://127.0.0.1:8765`

Start the server:

```bash
python -m navine.server
```

## Authentication

Routes under `/api/*` and `/v1/*` require authentication unless the client is localhost and `allow_localhost_without_auth` is enabled (default in `configs/api.yaml`).

Headers:

```
Authorization: Bearer YOUR_API_KEY
X-Navine-API-Key: YOUR_API_KEY
```

Create API keys:

```bash
python -m navine.cli api-key create --name "my key"
python -m navine.cli api-key list
```

`GET /api/health` does not require authentication.

## Endpoints

### GET /api/health

Health check.

Response:

```json
{ "status": "ok" }
```

```bash
curl http://127.0.0.1:8765/api/health
```

### GET /api/info

System and model status.

Response includes `pytorch`, `cuda_available`, `gpu`, `models`, `local_only`, and `training_types`.

```bash
curl http://127.0.0.1:8765/api/info
```

### POST /api/text

Generate text from a prompt.

Request:

```json
{
  "prompt": "Explain local AI in one sentence.",
  "max_tokens": 256,
  "temperature": 0.7
}
```

Response:

```json
{ "text": "..." }
```

```bash
curl -X POST http://127.0.0.1:8765/api/text \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello from Navine AI"}'
```

### POST /api/chat

Conversational chat with history and code intent detection.

Request:

```json
{
  "message": "Write a python function to reverse a string",
  "history": [],
  "max_tokens": 512,
  "temperature": 0.7,
  "use_rag": true
}
```

Response:

```json
{
  "text": "Here is the python code:\n\n```python\ndef reverse_string(s): ...",
  "message": { "role": "assistant", "content": "..." },
  "is_code": true
}
```

```bash
curl -X POST http://127.0.0.1:8765/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Write a fibonacci function in Python"}'
```

### POST /api/code

Generate code for a task.

Request:

```json
{
  "task": "Write a function to reverse a string",
  "language": "python"
}
```

Supported languages: `python`, `javascript`, `typescript`, `rust`, `go`, `java`, `cpp`

Response:

```json
{
  "code": "def reverse_string(s: str) -> str:\n    return s[::-1]",
  "language": "python"
}
```

```bash
curl -X POST http://127.0.0.1:8765/api/code \
  -H "Content-Type: application/json" \
  -d '{"task":"Write a fibonacci function","language":"python"}'
```

### POST /api/image

Generate an image from a prompt. Uses local diffusion with procedural fallback for recognizable simple prompts.

Request:

```json
{
  "prompt": "create me a blue moon",
  "num_steps": 100,
  "guidance_scale": 3.5,
  "seed": 42,
  "enhance": true
}
```

Response:

```json
{
  "path": "outputs/image/generated.png",
  "mime": "image/png",
  "data_base64": "...",
  "enhanced_prompt": "a blue moon in night sky, digital art, high quality..."
}
```

```bash
curl -X POST http://127.0.0.1:8765/api/image \
  -H "Content-Type: application/json" \
  -d '{"prompt":"create me a blue moon","num_steps":100}'
```

### POST /api/video

Generate a short animated GIF.

Request:

```json
{
  "prompt": "waves on a beach",
  "num_frames": 16,
  "fps": 12,
  "seed": 42
}
```

Response:

```json
{
  "path": "outputs/video/generated.gif",
  "mime": "image/gif",
  "data_base64": "..."
}
```

```bash
curl -X POST http://127.0.0.1:8765/api/video \
  -H "Content-Type: application/json" \
  -d '{"prompt":"waves on a beach","num_frames":16,"fps":12}'
```

### POST /v1/chat/completions

OpenAI-compatible chat completions endpoint.

Request:

```json
{
  "model": "navine-text",
  "messages": [
    { "role": "user", "content": "Hello" }
  ],
  "max_tokens": 256,
  "temperature": 0.7
}
```

Response follows OpenAI chat completion format with `choices`, `message`, and `usage` fields.

```bash
curl -X POST http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"model":"navine-text","messages":[{"role":"user","content":"Hello"}]}'
```

Streaming (`stream: true`) is not supported.

## Additional Routes

- `GET /api/train/types` - List available training types
- `GET /api/autolearn/status` - Autolearn engine status
- `GET /api/marathon/status` - Marathon learning daemon status

## Error Codes

| Code | Meaning |
|------|---------|
| 400 | Invalid request |
| 401 | Invalid or missing API key |
| 429 | Rate limit exceeded |
| 500 | Internal server error |
| 503 | Model not trained or unavailable |

## Web UI

- Main UI: `http://127.0.0.1:8765/`
- API docs: `http://127.0.0.1:8765/docs`
- Text, Image, and Video tabs in the web UI (code generation available via chat)

## CLI Equivalents

```bash
python -m navine.cli chat
python -m navine.cli code "Write a fibonacci function" --language python
python -m navine.cli image "create me a blue moon" --steps 100
```
