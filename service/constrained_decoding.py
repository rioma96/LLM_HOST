"""Constrained decoding with Outlines for strict JSON schema enforcement."""

import json
from typing import Any, Dict, Optional

try:
    from outlines import models as outlines_models
    from outlines import generate as outlines_generate
    OUTLINES_AVAILABLE = True
except ImportError:
    OUTLINES_AVAILABLE = False


# JSON schemas for strict output validation
VALIDATION_RESPONSE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "samples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sample_id": {"type": "string"},
                    "keep": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "delete": {"type": "boolean"},
                },
                "required": ["sample_id", "keep", "delete"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["samples"],
    "additionalProperties": False,
}


GENERATION_RESPONSE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "samples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sample_id": {"type": "string"},
                    "generated_text": {"type": "string"},
                    "relation": {"type": "string"},
                },
                "required": ["sample_id", "generated_text", "relation"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["samples"],
    "additionalProperties": False,
}


def get_constrained_generator(mode: str):
    """
    Get an Outlines JSON generator for the given mode.
    
    Args:
        mode: "review" or "generation"
    
    Returns:
        Callable that enforces JSON schema, or None if Outlines unavailable
    """
    if not OUTLINES_AVAILABLE:
        return None
    
    if mode == "review":
        schema = VALIDATION_RESPONSE_SCHEMA
    elif mode == "generation":
        schema = GENERATION_RESPONSE_SCHEMA
    else:
        return None
    
    try:
        return outlines_generate.json(schema)
    except Exception as exc:
        print(f"Failed to create constrained generator for mode {mode}: {exc}")
        return None


def validate_constrained_output(response: str, mode: str) -> bool:
    """
    Validate that response matches the strict JSON schema for given mode.
    
    Args:
        response: Raw LLM response string
        mode: "review" or "generation"
    
    Returns:
        True if valid, False otherwise
    """
    try:
        data = json.loads(response)
        
        if mode == "review":
            schema = VALIDATION_RESPONSE_SCHEMA
        elif mode == "generation":
            schema = GENERATION_RESPONSE_SCHEMA
        else:
            return False
        
        # Basic schema validation
        if not isinstance(data, dict) or "samples" not in data:
            return False
        
        samples = data.get("samples")
        if not isinstance(samples, list):
            return False
        
        for sample in samples:
            if not isinstance(sample, dict):
                return False
            
            if mode == "review":
                if not all(k in sample for k in ["sample_id", "keep", "delete"]):
                    return False
                if not isinstance(sample.get("keep"), list):
                    return False
                if not isinstance(sample.get("delete"), bool):
                    return False
                
                if sample.get("delete") and len(sample.get("keep")) > 0:
                    return False
                if not sample.get("delete") and len(sample.get("keep")) == 0:
                    return False
            elif mode == "generation":
                if not all(k in sample for k in ["sample_id", "generated_text", "relation"]):
                    return False
                if not isinstance(sample.get("generated_text"), str):
                    return False
        
        return True
    except Exception:
        return False


__all__ = [
    "VALIDATION_RESPONSE_SCHEMA",
    "GENERATION_RESPONSE_SCHEMA",
    "get_constrained_generator",
    "validate_constrained_output",
    "OUTLINES_AVAILABLE",
]
