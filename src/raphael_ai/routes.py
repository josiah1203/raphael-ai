"""API routes for raphael-ai."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["raphael-ai"])


@router.get("")
def list_root() -> dict[str, str]:
  return {"service": "raphael-ai", "status": "stub"}
