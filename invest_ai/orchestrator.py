"""Request-level orchestrator service and audit telemetry."""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict

from pydantic import ValidationError

from invest_ai.agents import create_gemini_agents
from invest_ai.compliance import ComplianceGuardrailEngine
from invest_ai.contracts import (
    AgentDecisionOutput,
    CustomerQueryInput,
)
from invest_ai.graph import InvestAIGraph
from invest_ai.rag import QdrantRagIndex

logger = logging.getLogger("LBG-InvestAI")


class InvestAIAgentOrchestrator:
    """Validates requests, runs the agent graph, and records summary telemetry."""

    def __init__(self, gcp_project_id: str = "lloyds-invest-ai-prod"):
        self.project_id = gcp_project_id
        router, advisor, model_name = create_gemini_agents()
        self.model_name = model_name
        self.guardrail_engine = ComplianceGuardrailEngine()
        self.rag_index = QdrantRagIndex()
        self.graph = InvestAIGraph(
            router=router,
            advisor=advisor,
            compliance=self.guardrail_engine,
            rag_index=self.rag_index,
        ).compiled
        logger.info(
            "Initialized InvestAIAgentOrchestrator with Gemini model %s",
            self.model_name,
        )

    def process_customer_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        start_time = datetime.now().astimezone()
        start_clock = time.perf_counter()

        try:
            validated_input = CustomerQueryInput(**payload)
        except ValidationError as exc:
            logger.error("Input validation error: %s", exc.json())
            raise ValueError(f"Invalid request format: {exc}") from exc

        customer_context = validated_input.context
        state = self.graph.invoke(
            {
                "query_text": validated_input.query_text,
                "context": customer_context.model_dump(),
            }
        )
        if state.get("rejection_message"):
            return {
                "status": "REJECTED",
                "message": state["rejection_message"],
            }

        decision = AgentDecisionOutput(
            customer_id=customer_context.customer_id,
            original_query=validated_input.query_text,
            generated_guidance=state["generated_guidance"],
            is_compliant=state["is_compliant"],
            compliance_flags=state["compliance_flags"],
            processing_timestamp=start_time.isoformat(),
            execution_latency_ms=round(
                (time.perf_counter() - start_clock) * 1000.0, 2
            ),
            sources=state["retrieved_sources"],
        )
        self._log_audit_telemetry(decision, customer_context.age_band)
        return decision.model_dump()

    @staticmethod
    def _log_audit_telemetry(
        decision: AgentDecisionOutput, age_band: str
    ) -> None:
        log_payload = {
            "event": "AGENT_DECISION",
            "customer_id": decision.customer_id,
            "age_band": age_band,
            "is_compliant": decision.is_compliant,
            "flags_count": len(decision.compliance_flags),
            "latency_ms": decision.execution_latency_ms,
            "timestamp": decision.processing_timestamp,
        }
        logger.info("AUDIT_TELEMETRY: %s", json.dumps(log_payload))
