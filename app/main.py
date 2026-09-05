"""
PayBack AI - FastAPI application entrypoint.

Wires together the /transactions, /decisions, and /metrics routers on top
of the core pipeline built in earlier stages, plus a basic /health check.
"""
from fastapi import FastAPI

from api.routes_decisions import router as decisions_router
from api.routes_metrics import router as metrics_router
from api.routes_transactions import router as transactions_router

app = FastAPI(
    title="PayBack AI",
    description="Synthetic revenue-recovery decision system (Development Version).",
    version="0.1.0",
)

app.include_router(transactions_router)
app.include_router(decisions_router)
app.include_router(metrics_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Basic liveness check for the API."""
    return {"status": "ok", "service": "payback-ai"}
