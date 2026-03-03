"""
General Routes for AI Resume Analyzer

Handles health checks, readiness checks, and serves the React SPA in production.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.services.model_manager import model_manager

router = APIRouter()


@router.get("/health")
def health():
    """Health check endpoint"""
    return "ok"


@router.get("/ready")
def ready_check():
    """Readiness check endpoint - checks if models are loaded"""
    is_ready = model_manager.is_loaded()
    status_code = 200 if is_ready else 503
    return JSONResponse(
        content={
            'status': 'ready' if is_ready else 'loading',
            'models_loaded': is_ready
        },
        status_code=status_code
    )
