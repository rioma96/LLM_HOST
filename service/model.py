import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationConfig:
    model_id: str
    max_new_tokens: int
    temperature: float
    top_p: float
    load_in_4bit: bool


class RelationOracleModel:
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)

        model_kwargs: Dict[str, Any] = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }

        if self.config.load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                **model_kwargs,
            )
        except Exception:
            model_kwargs.pop("load_in_4bit", None)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                **model_kwargs,
            )

    def infer(self, prompt: str) -> Dict[str, Any]:
        text = self.generate_text(prompt)
        parsed = self._safe_parse_output(text)
        parsed["raw_output"] = text
        return parsed

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        self.load()
        assert self.tokenizer is not None
        assert self.model is not None

        effective_max_new_tokens = max_new_tokens or self.config.max_new_tokens
        effective_temperature = self.config.temperature if temperature is None else temperature
        effective_top_p = self.config.top_p if top_p is None else top_p

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=effective_max_new_tokens,
                do_sample=effective_temperature > 0,
                temperature=effective_temperature,
                top_p=effective_top_p,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    @staticmethod
    def _safe_parse_output(text: str) -> Dict[str, Any]:
        json_candidate = text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_candidate = match.group(0)

        try:
            data = json.loads(json_candidate)
            relation_present = bool(data.get("relation_present", False))
            confidence = float(data.get("confidence", 0.5))
            reason_short = str(data.get("reason_short", "No reason provided."))
        except Exception:
            lowered = text.lower()
            relation_present = "true" in lowered and "false" not in lowered
            confidence = 0.5
            reason_short = "Output non pienamente strutturato; fallback parser applicato."

        confidence = max(0.0, min(1.0, confidence))
        return {
            "relation_present": relation_present,
            "confidence": confidence,
            "reason_short": reason_short,
        }


def from_env() -> GenerationConfig:
    return GenerationConfig(
        model_id=os.getenv("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2"),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "96")),
        temperature=float(os.getenv("TEMPERATURE", "0.1")),
        top_p=float(os.getenv("TOP_P", "0.9")),
        load_in_4bit=os.getenv("LOAD_IN_4BIT", "true").lower() == "true",
    )
