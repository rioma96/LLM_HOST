from typing import List, Optional

from pydantic import BaseModel, Field


class ModelStatusResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    loaded_in_memory: bool


class ValidationSample(BaseModel):
    model_config = {"extra": "ignore"}

    sample_id: str = Field(..., min_length=1)
    keep: Optional[List[str]] = None
    delete: bool = False
    notes: str = Field(default="")


class ValidationSamplesResponse(BaseModel):
    model_config = {"extra": "ignore"}

    samples: List[ValidationSample] = Field(default_factory=list)


class GenerationTriple(BaseModel):
    model_config = {"extra": "ignore"}

    head: str = Field(..., min_length=1)
    tail: str = Field(..., min_length=1)
    relation: str = Field(..., min_length=1)


class GenerationExample(BaseModel):
    model_config = {"extra": "ignore"}

    text: str = Field(default="")
    labels: List[str] = Field(default_factory=list)
    triple: GenerationTriple
    notes: str = Field(default="")


class GenerationExamplesResponse(BaseModel):
    model_config = {"extra": "ignore"}

    examples: List[GenerationExample] = Field(default_factory=list)
