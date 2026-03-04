"""
Routes package for AI Resume Analyzer

Imports and re-exports all route routers.
"""

from backend.routes.general import router as general_router
from backend.routes.upload import router as upload_router
from backend.routes.match import router as match_router
from backend.routes.ats import router as ats_router
from backend.routes.analyze import router as analyze_router

__all__ = ['general_router', 'upload_router', 'match_router', 'ats_router', 'analyze_router']
