"""Raphael service: raphael-ai."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from raphael_contracts.errors import ErrorResponse
from raphael_ai.routes import router

app = FastAPI(
    title="raphael-ai",
    description="AI understanding, copilot, agents, workflow generation",
    version="0.1.0",
    openapi_url="/v1/ai/openapi.json" if "/v1/ai" else "/openapi.json",
)

app.include_router(router, prefix="/v1/ai" if "/v1/ai" else "")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "raphael-ai"}


@app.exception_handler(Exception)
async def unhandled(_request, exc: Exception) -> JSONResponse:
    err = ErrorResponse(code="internal_error", message=str(exc))
    return JSONResponse(status_code=500, content=err.model_dump())
