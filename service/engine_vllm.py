"""vLLM-based model engine with Outlines constrained decoding support."""

import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for vLLM engine (compatible with model_shared.py)."""
    backend: str
    model_id: str
    max_new_tokens: int
    temperature: float
    top_p: float
    load_in_8bit: bool = False  # Note: vLLM handles quantization differently
    load_in_4bit: bool = False


class VLLMEngine:
    """vLLM-based inference engine with optional Outlines constrained decoding."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self.engine = None
        self.tokenizer = None
        self._lock = threading.Lock()
        self._constrained_generators = {}
        self._enable_constrained_decoding = True

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except ValueError:
            return default

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except ValueError:
            return default

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def load(self) -> None:
        """Initialize vLLM engine and tokenizer."""
        if self.engine is not None:
            return

        try:
            from vllm import LLM
            from vllm.sampling_params import SamplingParams
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "vLLM is required for VLLMEngine. Install with: pip install vllm outlines"
            ) from exc

        # Initialize vLLM with appropriate quantization
        engine_kwargs = {
            "model": self.config.model_id,
            "dtype": "float16",
            "gpu_memory_utilization": self._env_float("GPU_MEMORY_UTILIZATION", 0.60),
            "max_model_len": self._env_int("MAX_MODEL_LEN", 2048),
            "max_num_seqs": self._env_int("MAX_NUM_SEQS", 1),
            "enforce_eager": self._env_bool("VLLM_ENFORCE_EAGER", True),
        }

        # Match the legacy startup command: 4-bit should use bitsandbytes, not AWQ.
        if self.config.load_in_4bit:
            engine_kwargs["quantization"] = "bitsandbytes"
            engine_kwargs["load_format"] = "bitsandbytes"
        elif self.config.load_in_8bit:
            # 8-bit support varies by vLLM version; fallback to float16
            logger.warning("8-bit quantization not directly supported in vLLM; using float16")

        try:
            self.engine = LLM(**engine_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info(f"vLLM engine loaded: {self.config.model_id}")
        except Exception as exc:
            raise RuntimeError(f"Failed to load vLLM engine: {exc}") from exc

    def is_loaded(self) -> bool:
        """Return True when the vLLM engine has already been initialized."""
        return self.engine is not None

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Generate text from a plain prompt."""
        self.load()
        assert self.engine is not None

        from vllm.sampling_params import SamplingParams, StructuredOutputsParams

        effective_max_new_tokens = max_new_tokens or self.config.max_new_tokens
        effective_temperature = self.config.temperature if temperature is None else temperature
        effective_top_p = self.config.top_p if top_p is None else top_p

        sampling_params = SamplingParams(
            max_tokens=effective_max_new_tokens,
            temperature=effective_temperature,
            top_p=effective_top_p,
        )

        acquired = self._lock.acquire(timeout=180)
        if not acquired:
            raise RuntimeError("vLLM generation lock timeout – server is overloaded")

        try:
            outputs = self.engine.generate([prompt], sampling_params, use_tqdm=False)
            generated_text = outputs[0].outputs[0].text
            return generated_text.strip()
        finally:
            self._lock.release()

    def generate_chat_text(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Generate text from a chat-format prompt."""
        self.load()
        assert self.engine is not None
        assert self.tokenizer is not None

        from vllm.sampling_params import SamplingParams, StructuredOutputsParams

        effective_max_new_tokens = max_new_tokens or self.config.max_new_tokens
        effective_temperature = self.config.temperature if temperature is None else temperature
        effective_top_p = self.config.top_p if top_p is None else top_p

        # Apply chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as exc:
                logger.warning(f"apply_chat_template failed, using plain text: {exc}")
                prompt = self._fallback_messages_to_prompt(messages)
        else:
            prompt = self._fallback_messages_to_prompt(messages)

        sampling_params = SamplingParams(
            max_tokens=effective_max_new_tokens,
            temperature=effective_temperature,
            top_p=effective_top_p,
        )

        acquired = self._lock.acquire(timeout=180)
        if not acquired:
            raise RuntimeError("vLLM generation lock timeout – server is overloaded")

        try:
            outputs = self.engine.generate([prompt], sampling_params, use_tqdm=False)
            generated_text = outputs[0].outputs[0].text
            return generated_text.strip()
        finally:
            self._lock.release()

    def generate_chat_text_with_constraint(
        self,
        messages: List[Dict[str, Any]],
        constraint_schema: Optional[Dict[str, Any]] = None,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Generate text from chat messages with optional JSON schema constraint."""
        if not constraint_schema or not self._enable_constrained_decoding:
            # Fallback to unconstrained generation
            return self.generate_chat_text(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        self.load()
        assert self.engine is not None
        assert self.tokenizer is not None

        from vllm.sampling_params import SamplingParams, StructuredOutputsParams

        effective_max_new_tokens = max_new_tokens or self.config.max_new_tokens
        effective_temperature = self.config.temperature if temperature is None else temperature
        effective_top_p = self.config.top_p if top_p is None else top_p

        # Build chat prompt
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as exc:
                logger.warning(f"apply_chat_template failed: {exc}")
                prompt = self._fallback_messages_to_prompt(messages)
        else:
            prompt = self._fallback_messages_to_prompt(messages)

        sampling_params = SamplingParams(
            max_tokens=effective_max_new_tokens,
            temperature=effective_temperature,
            top_p=effective_top_p,
            structured_outputs=StructuredOutputsParams(json=constraint_schema),
        )
        acquired = self._lock.acquire(timeout=180)
        if not acquired:
            raise RuntimeError("vLLM generation lock timeout")

        try:
            outputs = self.engine.generate([prompt], sampling_params, use_tqdm=False)
            generated_text = outputs[0].outputs[0].text
            return generated_text.strip()
        finally:
            self._lock.release()

    @staticmethod
    def _fallback_messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
        """Convert chat messages to plain text prompt."""
        lines: List[str] = []
        for msg in messages:
            role = str(msg.get("role", "user")).upper()
            content = str(msg.get("content", ""))
            lines.append(f"[{role}]\n{content}")
        return "\n\n".join(lines).strip()


__all__ = ["VLLMEngine", "GenerationConfig"]
