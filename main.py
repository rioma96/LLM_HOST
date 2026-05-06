import os
import json
import time
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from starlette.responses import Response as StarletteResponse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from service.constrained_decoding import (
    GENERATION_RESPONSE_SCHEMA,
    VALIDATION_RESPONSE_SCHEMA,
    validate_constrained_output,
)
from service.model_factory import create_oracle_from_env_with_engine
from service.schemas import ModelStatusResponse


load_dotenv()
# setup simple request logger (file + stdout)
_logs_dir = Path(__file__).parent / "logs"
_logs_dir.mkdir(parents=True, exist_ok=True)
_log_file = _logs_dir / "requests.log"

logger = logging.getLogger("llm_requests")
logger.setLevel(logging.INFO)
if not logger.handlers:
    # rotate around 1GB per file, keep 1 backup (current + 1 = ~2GB max)
    fh = RotatingFileHandler(str(_log_file), maxBytes=1 * 1024 * 1024 * 1024, backupCount=1, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

app = FastAPI(title="LLM Relation Oracle", version="0.1.0")
oracle = create_oracle_from_env_with_engine()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    try:
        body_bytes = await request.body()
        body_text = body_bytes.decode("utf-8", errors="replace") if body_bytes else ""
        if len(body_text) > 2000:
            body_text = body_text[:2000] + "...(truncated)"
    except Exception as e:
        body_text = f"<error reading body: {e}>"

    client = request.client.host if request.client else "unknown"
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Error handling request")
        raise

    resp_body_bytes = b""
    try:
        async for chunk in response.body_iterator:
            resp_body_bytes += chunk
    except Exception:
        try:
            resp_body_bytes = getattr(response, "body", b"") or b""
        except Exception:
            resp_body_bytes = b""

    max_capture = 5 * 1024 * 1024
    resp_text = resp_body_bytes.decode("utf-8", errors="replace")
    if len(resp_text) > max_capture:
        resp_text = resp_text[:max_capture] + "...(truncated)"

    process_time = (time.time() - start) * 1000.0
    log_entry = {
        "ts": int(time.time()),
        "client": client,
        "method": request.method,
        "path": str(request.url.path),
        "status_code": response.status_code,
        "process_ms": round(process_time, 2),
        "request_body": body_text,
        "response_body": resp_text,
    }
    try:
        logger.info(json.dumps(log_entry, ensure_ascii=False))
    except Exception:
        logger.info(str(log_entry))

    new_response = StarletteResponse(content=resp_body_bytes, status_code=response.status_code)
    for k, v in response.headers.items():
        new_response.headers[k] = v
    return new_response


def _check_openai_api_key(authorization: Optional[str]) -> None:
    expected = os.getenv("OPENAI_API_KEY", "").strip()
    if not expected:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    provided = authorization.split(" ", 1)[1].strip()
    if provided != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_id": oracle.config.model_id}


@app.get("/v1/model/status", response_model=ModelStatusResponse)
def model_status() -> ModelStatusResponse:
    return ModelStatusResponse(
        model_id=oracle.config.model_id,
        loaded_in_memory=oracle.is_loaded(),
    )


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": oracle.config.model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    payload: Dict[str, Any],
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _check_openai_api_key(authorization)

    messages = payload.get("messages", [])
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")

    mode = str(payload.get("mode", "review")).strip().lower()
    if mode not in {"review", "generation"}:
        raise HTTPException(status_code=400, detail="mode must be one of: review, generation")

    max_tokens = int(payload.get("max_tokens", oracle.config.max_new_tokens))
    temperature = float(payload.get("temperature", oracle.config.temperature))
    top_p = float(payload.get("top_p", oracle.config.top_p))

    # server-side max_tokens cap per mode
    max_tokens_review = 256
    max_tokens_generation = 512
    if mode == "review":
        max_tokens = min(max_tokens, max_tokens_review)
    else:
        max_tokens = min(max_tokens, max_tokens_generation)

    # deterministic validation for review, bounded sampling for generation
    if mode == "review":
        if temperature > 0.01:
            logger.warning(
                "Review mode requested temp=%s; forcing temp=0.0 for deterministic validation",
                temperature,
            )
        temperature = 0.0
        top_p = 1.0
    else:
        if temperature > 0.7:
            logger.warning("Generation mode requested temp=%s; capping at 0.7", temperature)
            temperature = 0.7
        elif temperature <= 0.0:
            temperature = 0.3

    schema = VALIDATION_RESPONSE_SCHEMA if mode == "review" else GENERATION_RESPONSE_SCHEMA

    try:
        if hasattr(oracle, "generate_chat_text_with_constraint"):
            raw_output_text = await oracle.generate_chat_text_with_constraint(
                messages,
                constraint_schema=schema,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            raw_output_text = await oracle.generate_chat_text(
                messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
    except Exception as err:
        raise HTTPException(
            status_code=503,
            detail=f"Model generation failed: {err}",
        ) from err

    if not isinstance(raw_output_text, str) or not raw_output_text.strip():
        raise HTTPException(status_code=502, detail="Model returned empty response")

    if not validate_constrained_output(raw_output_text, mode):
        raise HTTPException(
            status_code=502,
            detail="Model returned invalid JSON for strict constrained schema",
        )

    normalized_payload = json.loads(raw_output_text)
    output_text = json.dumps(normalized_payload, ensure_ascii=False)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model_name = oracle.config.model_id
    prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
    prompt_tokens = max(1, prompt_chars // 4)
    completion_tokens = max(1, len(output_text) // 4)

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=False)
