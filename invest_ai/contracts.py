"""Validated request, response, and LangGraph state contracts."""

from typing import Any, Dict, List, Literal, TypedDict

from pydantic import BaseModel, Field


class CustomerContext(BaseModel):
    customer_id: str = Field(..., description="Anonymized Customer Identifier")
    age_band: str = Field(
        ...,
        description="Demographic age band e.g., '18-24', '25-34', '35-50', '51-65', '65+'",
    )
    investor_experience: str = Field(
        ..., description="'Novice', 'Intermediate', or 'Advanced'"
    )
    has_pension: bool = Field(default=True)


class CustomerQueryInput(BaseModel):
    context: CustomerContext
    query_text: str = Field(
        ..., min_length=5, description="Raw query text from retail customer"
    )


class RouterDecision(BaseModel):
    intent: Literal["INVESTMENT_GUIDANCE", "GENERAL_INQUIRY"]


class AgentState(TypedDict, total=False):
    query_text: str
    context: Dict[str, Any]
    intent: str
    retrieved_passages: List[Dict[str, Any]]
    retrieved_sources: List[Dict[str, Any]]
    raw_draft: str
    is_compliant: bool
    compliance_flags: List[str]
    generated_guidance: str
    rejection_message: str


class AgentDecisionOutput(BaseModel):
    customer_id: str
    original_query: str
    generated_guidance: str
    is_compliant: bool
    compliance_flags: List[str]
    processing_timestamp: str
    execution_latency_ms: float
    sources: List[Dict[str, Any]] = Field(default_factory=list)
