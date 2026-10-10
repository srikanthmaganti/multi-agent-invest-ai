"""LangGraph workflow that wires router, advisor, and compliance nodes."""

from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from invest_ai.agents import AdvisorAgent, RouterAgent
from invest_ai.compliance import ComplianceGuardrailEngine
from invest_ai.contracts import AgentState, CustomerContext
from invest_ai.rag import QdrantRagIndex, RetrievedPassage


class InvestAIGraph:
    """Build and expose the request-processing LangGraph."""

    def __init__(
        self,
        router: RouterAgent,
        advisor: AdvisorAgent,
        compliance: ComplianceGuardrailEngine,
        rag_index: QdrantRagIndex,
    ):
        self.router = router
        self.advisor = advisor
        self.compliance = compliance
        self.rag_index = rag_index
        self.compiled = self._build()

    def _router_node(self, state: AgentState) -> Dict[str, str]:
        return {"intent": self.router.classify(state["query_text"])}

    def _advisor_node(self, state: AgentState) -> Dict[str, str]:
        context = CustomerContext.model_validate(state["context"])
        return {
            "raw_draft": self.advisor.respond(
                state["query_text"],
                context,
                [
                    RetrievedPassage(**passage)
                    for passage in state["retrieved_passages"]
                ],
            )
        }

    def _retrieval_node(self, state: AgentState) -> Dict[str, Any]:
        passages = self.rag_index.retrieve(state["query_text"])
        return {
            "retrieved_passages": [
                {
                    "citation_id": passage.citation_id,
                    "text": passage.text,
                    "document_title": passage.document_title,
                    "publication": passage.publication,
                    "source_url": passage.source_url,
                    "page_numbers": passage.page_numbers,
                    "section": passage.section,
                    "score": passage.score,
                }
                for passage in passages
            ],
            "retrieved_sources": [passage.citation() for passage in passages],
        }

    def _compliance_node(self, state: AgentState) -> Dict[str, Any]:
        is_compliant, flags, final_response = self.compliance.evaluate_response(
            state["raw_draft"]
        )
        return {
            "is_compliant": is_compliant,
            "compliance_flags": flags,
            "generated_guidance": final_response,
        }

    @staticmethod
    def _out_of_scope_node(state: AgentState) -> Dict[str, str]:
        if state["intent"] != "GENERAL_INQUIRY":
            raise RuntimeError("Out-of-scope node received an unexpected intent.")
        return {
            "rejection_message": "Query out of scope for Investment Guidance Agent."
        }

    @staticmethod
    def _route_after_classification(state: AgentState) -> str:
        return (
            "advisor"
            if state["intent"] == "INVESTMENT_GUIDANCE"
            else "out_of_scope"
        )

    def _build(self) -> Any:
        workflow = StateGraph(AgentState)
        workflow.add_node("router", self._router_node)
        workflow.add_node("retrieval", self._retrieval_node)
        workflow.add_node("advisor", self._advisor_node)
        workflow.add_node("compliance", self._compliance_node)
        workflow.add_node("out_of_scope", self._out_of_scope_node)

        workflow.add_edge(START, "router")
        workflow.add_conditional_edges(
            "router",
            self._route_after_classification,
            {
                "advisor": "retrieval",
                "out_of_scope": "out_of_scope",
            },
        )
        workflow.add_edge("retrieval", "advisor")
        workflow.add_edge("advisor", "compliance")
        workflow.add_edge("compliance", END)
        workflow.add_edge("out_of_scope", END)
        return workflow.compile()
