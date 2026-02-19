from typing import List

from pydantic import BaseModel, Field


class RelationRequest(BaseModel):
    sentence: str = Field(..., min_length=3)
    entity_1: str = Field(..., min_length=1)
    entity_2: str = Field(..., min_length=1)
    relation_name: str = Field(..., min_length=1)
    relation_description: str = Field(default="")


class RelationResponse(BaseModel):
    relation_present: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason_short: str
    raw_output: str


class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=3)


class PromptResponse(BaseModel):
    output_text: str


class ModelStatusResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    loaded_in_memory: bool


class ValidationSample(BaseModel):
    model_config = {"extra": "ignore"}

    sample_id: str = Field(..., min_length=1)
    keep: List[str] = Field(default_factory=list)
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
