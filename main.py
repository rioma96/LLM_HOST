import os
import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from starlette.responses import Response as StarletteResponse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from service.model import RelationOracleModel, from_env
from service.prompting import build_relation_prompt
from service.schemas import (
    GenerationExamplesResponse,
    ModelStatusResponse,
    PromptRequest,
    PromptResponse,
    RelationRequest,
    RelationResponse,
    ValidationSamplesResponse,
)


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
oracle = RelationOracleModel(from_env())


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

    # capture response body (stream) and reconstruct response
    resp_body_bytes = b""
    try:
        async for chunk in response.body_iterator:
            resp_body_bytes += chunk
    except Exception as e:
        # if body_iterator isn't available or errors, fallback
        try:
            resp_body_bytes = getattr(response, "body", b"") or b""
        except Exception:
            resp_body_bytes = b""

    # limit per-request memory footprint while still logging; cap at 5MB per side
    MAX_CAPTURE = 5 * 1024 * 1024
    resp_text = resp_body_bytes.decode("utf-8", errors="replace")
    if len(resp_text) > MAX_CAPTURE:
        resp_text = resp_text[:MAX_CAPTURE] + "...(truncated)"

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

    # rebuild response for the client
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


def _messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in messages:
        role = str(item.get("role", "user")).upper()
        content = str(item.get("content", ""))
        lines.append(f"[{role}]\n{content}")
    return "\n\n".join(lines).strip()


def _strip_assistant_prefix(text: str) -> str:
    return re.sub(r"^\s*\[?ASSISTANT\]?\s*:?\s*", "", text.strip(), flags=re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1).strip()
    return stripped


def _try_parse_json_value(text: str) -> Optional[Any]:
    decoder = json.JSONDecoder()
    no_prefix = _strip_assistant_prefix(text)
    candidates = [text.strip(), _strip_code_fences(text), no_prefix, _strip_code_fences(no_prefix)]

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            pass

        for idx, char in enumerate(candidate):
            if char not in "[{":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[idx:])
                return parsed
            except Exception:
                continue

    return None


def _collect_objects_with_keys(data: Any, required_keys: List[str]) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        if all(key in data for key in required_keys):
            found.append(data)
        for value in data.values():
            found.extend(_collect_objects_with_keys(value, required_keys))
    elif isinstance(data, list):
        for item in data:
            found.extend(_collect_objects_with_keys(item, required_keys))
    return found


def _extract_input_sample_ids(messages: List[Dict[str, Any]]) -> List[str]:
    sample_ids: List[str] = []
    seen: set[str] = set()
    pattern = re.compile(r'"sample_id"\s*:\s*"([^"\\]+)"')

    for item in messages:
        content = str(item.get("content", ""))

        parsed = _try_parse_json_value(content)
        if parsed is not None:
            for sample in _collect_objects_with_keys(parsed, ["sample_id"]):
                sample_id = str(sample.get("sample_id", "")).strip()
                if sample_id and sample_id not in seen:
                    seen.add(sample_id)
                    sample_ids.append(sample_id)

        for match in pattern.finditer(content):
            sample_id = match.group(1).strip()
            if sample_id and sample_id not in seen:
                seen.add(sample_id)
                sample_ids.append(sample_id)

    return sample_ids


def _extract_input_generation_triples(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    triples: List[Dict[str, str]] = []
    seen: set[Tuple[str, str, str]] = set()
    pattern = re.compile(
        r'"head"\s*:\s*"([^"\\]+)".*?"relation"\s*:\s*"([^"\\]+)".*?"tail"\s*:\s*"([^"\\]+)"',
        re.DOTALL,
    )

    for item in messages:
        content = str(item.get("content", ""))
        parsed = _try_parse_json_value(content)
        if parsed is not None:
            for triple_obj in _collect_objects_with_keys(parsed, ["head", "tail", "relation"]):
                triple = {
                    "head": str(triple_obj.get("head", "")).strip(),
                    "tail": str(triple_obj.get("tail", "")).strip(),
                    "relation": str(triple_obj.get("relation", "")).strip(),
                }
                key = (triple["head"], triple["tail"], triple["relation"])
                if all(key) and key not in seen:
                    seen.add(key)
                    triples.append(triple)

        for match in pattern.finditer(content):
            head = match.group(1).strip()
            relation = match.group(2).strip()
            tail = match.group(3).strip()
            key = (head, tail, relation)
            if all(key) and key not in seen:
                seen.add(key)
                triples.append({"head": head, "tail": tail, "relation": relation})

    return triples


def _safe_review_fallback_payload(sample_ids: List[str]) -> Dict[str, Any]:
    ids = sample_ids or ["<input-sample-id>"]
    return {
        "samples": [
            {
                "sample_id": sample_id,
                "keep": [],
                "delete": False,
                "notes": "server_normalization_fallback",
            }
            for sample_id in ids
        ]
    }


def _safe_generation_fallback_payload() -> Dict[str, Any]:
    return {"examples": []}


def _coerce_review_sample(item: Any, default_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None

    sample_id = str(item.get("sample_id") or default_id or "").strip()
    if not sample_id:
        return None

    keep_raw = item.get("keep", [])
    if isinstance(keep_raw, list):
        keep = [str(label) for label in keep_raw if str(label).strip()]
    elif keep_raw is None:
        keep = []
    else:
        keep = [str(keep_raw)]

    return {
        "sample_id": sample_id,
        "keep": keep,
        "delete": bool(item.get("delete", False)),
        "notes": str(item.get("notes", "")),
    }


def _triple_key(triple: Dict[str, str]) -> Tuple[str, str, str]:
    return (
        str(triple.get("head", "")).strip(),
        str(triple.get("tail", "")).strip(),
        str(triple.get("relation", "")).strip(),
    )


def _coerce_generation_example(
    item: Any,
    default_triple: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None

    triple_raw = item.get("triple")
    if not isinstance(triple_raw, dict):
        if all(k in item for k in ["head", "tail", "relation"]):
            triple_raw = {
                "head": item.get("head", ""),
                "tail": item.get("tail", ""),
                "relation": item.get("relation", ""),
            }
        else:
            triple_raw = default_triple or {}

    triple = {
        "head": str(triple_raw.get("head", "")).strip(),
        "tail": str(triple_raw.get("tail", "")).strip(),
        "relation": str(triple_raw.get("relation", "")).strip(),
    }
    if not all(_triple_key(triple)):
        return None

    labels_raw = item.get("labels")
    if isinstance(labels_raw, list):
        labels = [str(label) for label in labels_raw if str(label).strip()]
    elif labels_raw is None:
        labels = []
    elif labels_raw:
        labels = [str(labels_raw)]
    else:
        labels = []

    if not labels:
        labels = [triple["relation"]]

    return {
        "text": str(item.get("text", "")),
        "labels": labels,
        "triple": triple,
        "notes": str(item.get("notes", "")),
    }


def _normalize_review_payload(raw_text: str, input_sample_ids: List[str]) -> Tuple[Dict[str, Any], str]:
    parsed = _try_parse_json_value(raw_text)
    if parsed is None:
        return _safe_review_fallback_payload(input_sample_ids), "parse_failed_fallback"

    samples_raw: List[Any] = []
    if isinstance(parsed, dict):
        if isinstance(parsed.get("samples"), list):
            samples_raw = parsed["samples"]
        elif "sample_id" in parsed:
            samples_raw = [parsed]
    elif isinstance(parsed, list):
        samples_raw = parsed

    by_id: Dict[str, Dict[str, Any]] = {}
    for sample in samples_raw:
        coerced = _coerce_review_sample(sample)
        if not coerced:
            continue
        sample_id = coerced["sample_id"]
        if sample_id not in by_id:
            by_id[sample_id] = coerced

    normalized_samples: List[Dict[str, Any]] = []
    missing_count = 0
    if input_sample_ids:
        for sample_id in input_sample_ids:
            if sample_id in by_id:
                item = dict(by_id[sample_id])
                item["sample_id"] = sample_id
                normalized_samples.append(item)
            else:
                missing_count += 1
                normalized_samples.append(
                    {
                        "sample_id": sample_id,
                        "keep": [],
                        "delete": False,
                        "notes": "server_normalization_fallback",
                    }
                )
    else:
        normalized_samples = list(by_id.values())

    payload: Dict[str, Any] = {"samples": normalized_samples}

    try:
        validated = ValidationSamplesResponse.model_validate(payload)
    except Exception:
        return _safe_review_fallback_payload(input_sample_ids), "validation_failed_fallback"

    validated_payload = validated.model_dump()
    if input_sample_ids and len(validated_payload.get("samples", [])) != len(input_sample_ids):
        return _safe_review_fallback_payload(input_sample_ids), "cardinality_mismatch_fallback"

    if missing_count > 0:
        return validated_payload, "filled_missing_items"
    if isinstance(parsed, (dict, list)):
        return validated_payload, "normalized_or_wrapped"
    return validated_payload, "schema_valid"


def _normalize_generation_payload(
    raw_text: str,
    input_triples: List[Dict[str, str]],
) -> Tuple[Dict[str, Any], str]:
    parsed = _try_parse_json_value(raw_text)
    if parsed is None:
        return _safe_generation_fallback_payload(), "parse_failed_fallback"

    examples_raw: List[Any] = []
    if isinstance(parsed, dict):
        if isinstance(parsed.get("examples"), list):
            examples_raw = parsed["examples"]
        elif "text" in parsed and ("triple" in parsed or all(k in parsed for k in ["head", "tail", "relation"])):
            examples_raw = [parsed]
    elif isinstance(parsed, list):
        examples_raw = parsed

    by_triple: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for example in examples_raw:
        coerced = _coerce_generation_example(example)
        if not coerced:
            continue
        key = _triple_key(coerced["triple"])
        if key not in by_triple:
            by_triple[key] = coerced

    normalized_examples: List[Dict[str, Any]] = []
    missing_count = 0
    if input_triples:
        for triple in input_triples:
            key = _triple_key(triple)
            if key in by_triple:
                example = dict(by_triple[key])
                example["triple"] = {
                    "head": triple["head"],
                    "tail": triple["tail"],
                    "relation": triple["relation"],
                }
                if not example.get("labels"):
                    example["labels"] = [triple["relation"]]
                normalized_examples.append(example)
            else:
                missing_count += 1
                normalized_examples.append(
                    {
                        "text": "",
                        "labels": [triple["relation"]],
                        "triple": {
                            "head": triple["head"],
                            "tail": triple["tail"],
                            "relation": triple["relation"],
                        },
                        "notes": "server_normalization_fallback",
                    }
                )
    else:
        normalized_examples = list(by_triple.values())

    payload: Dict[str, Any] = {"examples": normalized_examples}

    try:
        validated = GenerationExamplesResponse.model_validate(payload)
    except Exception:
        return _safe_generation_fallback_payload(), "validation_failed_fallback"

    validated_payload = validated.model_dump()
    if input_triples and len(validated_payload.get("examples", [])) != len(input_triples):
        return _safe_generation_fallback_payload(), "cardinality_mismatch_fallback"

    if missing_count > 0:
        return validated_payload, "filled_missing_items"
    return validated_payload, "normalized_or_wrapped"


def _normalize_mode_output(
    *,
    mode: str,
    raw_text: str,
    input_sample_ids: List[str],
    input_triples: List[Dict[str, str]],
) -> Tuple[str, bool, str]:
    mode_value = (mode or "review").strip().lower()
    if mode_value == "generation":
        payload, reason = _normalize_generation_payload(raw_text, input_triples)
    else:
        payload, reason = _normalize_review_payload(raw_text, input_sample_ids)

    normalized_json = json.dumps(payload, ensure_ascii=False)
    compare_text = _strip_code_fences(_strip_assistant_prefix(raw_text)).strip()
    normalization_applied = normalized_json != compare_text
    return normalized_json, normalization_applied, reason


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_id": oracle.config.model_id}


@app.get("/v1/model/status", response_model=ModelStatusResponse)
def model_status() -> ModelStatusResponse:
    return ModelStatusResponse(
        model_id=oracle.config.model_id,
        loaded_in_memory=oracle.is_loaded(),
    )


@app.post("/v1/relation/predict", response_model=RelationResponse)
def predict_relation(payload: RelationRequest) -> RelationResponse:
    prompt = build_relation_prompt(
        sentence=payload.sentence,
        entity_1=payload.entity_1,
        entity_2=payload.entity_2,
        relation_name=payload.relation_name,
        relation_description=payload.relation_description,
    )
    try:
        result = oracle.infer(prompt)
    except Exception as err:
        raise HTTPException(
            status_code=503,
            detail=f"Model inference failed: {err}",
        ) from err
    return RelationResponse(**result)


@app.post("/v1/generate", response_model=PromptResponse)
def generate_from_prompt(payload: PromptRequest) -> PromptResponse:
    try:
        output_text = oracle.generate_text(payload.prompt)
    except Exception as err:
        raise HTTPException(
            status_code=503,
            detail=f"Model generation failed: {err}",
        ) from err
    return PromptResponse(output_text=output_text)


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
def chat_completions(
    payload: Dict[str, Any],
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _check_openai_api_key(authorization)

    messages = payload.get("messages", [])
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")

    mode = str(payload.get("mode", "review")).strip().lower()
    input_sample_ids = _extract_input_sample_ids(messages)
    input_triples = _extract_input_generation_triples(messages)
    prompt = _messages_to_prompt(messages)
    max_tokens = int(payload.get("max_tokens", oracle.config.max_new_tokens))
    temperature = float(payload.get("temperature", oracle.config.temperature))

    try:
        output_text = oracle.generate_text(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as err:
        raise HTTPException(
            status_code=503,
            detail=f"Model generation failed: {err}",
        ) from err

    output_text, normalization_applied, normalization_reason = _normalize_mode_output(
        mode=mode,
        raw_text=output_text,
        input_sample_ids=input_sample_ids,
        input_triples=input_triples,
    )

    logger.info(
        json.dumps(
            {
                "event": "normalization",
                "mode": mode,
                "normalization_applied": normalization_applied,
                "reason": normalization_reason,
            },
            ensure_ascii=False,
        )
    )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model_name = str(payload.get("model") or oracle.config.model_id)
    prompt_tokens = max(1, len(prompt) // 4)
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
