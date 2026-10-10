"""FastAPI routes for the local Invest AI orchestrator service."""

import logging
from functools import lru_cache
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

from invest_ai.agents import GeminiServiceError
from invest_ai.contracts import CustomerQueryInput
from invest_ai.orchestrator import InvestAIAgentOrchestrator
from invest_ai.rag import RagIndexNotReadyError

logger = logging.getLogger("LBG-InvestAI")
api = FastAPI(title="Invest AI Orchestrator", version="1.0.0")


@lru_cache(maxsize=1)
def get_orchestrator() -> InvestAIAgentOrchestrator:
    """Create one orchestrator instance for requests handled by this process."""
    return InvestAIAgentOrchestrator()


@api.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@api.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return field-level validation errors without echoing submitted values."""
    validation_errors = [
        {
            "loc": error["loc"],
            "msg": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors()
    ]
    logger.warning(
        "Request validation failed for %s: %s", request.url.path, validation_errors
    )
    return JSONResponse(
        status_code=422,
        content={"detail": validation_errors},
    )


@api.post("/api/chat")
def chat(
    request: CustomerQueryInput,
    orchestrator: InvestAIAgentOrchestrator = Depends(get_orchestrator),
) -> Dict[str, Any]:
    try:
        return orchestrator.process_customer_request(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except GeminiServiceError as exc:
        logger.error("Orchestrator request failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except RagIndexNotReadyError as exc:
        logger.error("RAG index is not ready: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("Orchestrator request failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="The orchestrator could not complete the request. Check server logs.",
        ) from exc
