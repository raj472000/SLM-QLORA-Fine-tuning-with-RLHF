from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)


class GenerateResponse(BaseModel):
    output: str
    safe: bool
    reason: str
    fallback_used: bool
    latency_ms: float