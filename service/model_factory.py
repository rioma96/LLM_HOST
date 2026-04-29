import logging
import os

from service.engine_vllm import GenerationConfig, VLLMEngine

logger = logging.getLogger(__name__)


def from_env() -> GenerationConfig:
    default_model_id = os.getenv("VLLM_MODEL_ID", "Qwen/Qwen2.5-14B-Instruct")
    model_id = os.getenv("MODEL_ID", default_model_id)

    load_in_8bit = os.getenv("LOAD_IN_8BIT", "false").lower() == "true"
    load_in_4bit = os.getenv("LOAD_IN_4BIT", "false").lower() == "true"

    if load_in_8bit and load_in_4bit:
        logger.warning(
            "Both LOAD_IN_8BIT and LOAD_IN_4BIT are true; "
            "8-bit takes priority — set LOAD_IN_4BIT=false to silence this warning."
        )
        load_in_4bit = False

    return GenerationConfig(
        backend="vllm",
        model_id=model_id,
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "96")),
        temperature=float(os.getenv("TEMPERATURE", "0.1")),
        top_p=float(os.getenv("TOP_P", "0.9")),
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
    )


def create_oracle_from_env():
    return create_oracle_from_env_with_engine()


def create_oracle_from_env_with_engine():
    """Create oracle with strict JSON constrained decoding backend (vLLM)."""
    config = from_env()
    logger.info("Using vLLM backend for %s", config.model_id)
    return VLLMEngine(config)
