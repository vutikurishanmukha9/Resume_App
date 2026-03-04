import os
import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

from backend.config import BASE_DIR, FRONTEND_DIR, UPLOAD_FOLDER
from backend.exceptions import ResumeAnalyzerError
from backend.services.model_manager import load_models_background
from backend.rate_limiter import limiter, rate_limiting_enabled
from backend.routes import general_router, upload_router, match_router, ats_router, analyze_router

# -------------------- LOGGING --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info(" AI Resume Analyzer Starting (FastAPI)...")
logger.info("=" * 60)


# -------------------- LIFESPAN (startup/shutdown) --------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - handles startup and shutdown events"""
    # Startup: load models synchronously so the server is fully ready
    # before accepting traffic.  Container health checks should use
    # /ready (not /health) to gate traffic until this completes.
    logger.info("Loading ML models (server will accept requests after this)...")
    load_models_background()
    logger.info("All models loaded — server ready")

    yield

    # Shutdown
    logger.info("Application shutting down...")


# -------------------- APP CREATION --------------------
app = FastAPI(
    title="AI Resume Analyzer",
    description="Analyze resumes, match against job descriptions, and calculate ATS scores",
    version="2.0.0",
    lifespan=lifespan
)

# -------------------- MIDDLEWARE --------------------
# CORS for split deployment (React frontend + FastAPI backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
if rate_limiting_enabled:
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    logger.info("Rate limiting enabled")


# -------------------- API ROUTES --------------------
app.include_router(general_router)
app.include_router(upload_router)
app.include_router(match_router)
app.include_router(ats_router)
app.include_router(analyze_router)


# -------------------- REACT FRONTEND (Production) --------------------
# Serve the React build in production (after `npm run build`)
react_dist_dir = str(FRONTEND_DIR / "dist")
if os.path.isdir(react_dist_dir):
    # Serve static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=os.path.join(react_dist_dir, "assets")), name="assets")
    logger.info(f"React production build mounted from: {react_dist_dir}")

    # SPA fallback: serve index.html for all non-API routes
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        """Serve the React SPA for any non-API route"""
        # Check if the requested file exists in dist
        file_path = os.path.join(react_dist_dir, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        # Otherwise serve index.html (SPA routing)
        return FileResponse(os.path.join(react_dist_dir, "index.html"))
else:
    logger.info("No React build found. Run 'cd frontend && npm run build' for production deployment.")
    logger.info("In development, start the React dev server: 'cd frontend && npm run dev'")


# -------------------- ERROR HANDLERS --------------------
@app.exception_handler(ResumeAnalyzerError)
async def handle_resume_analyzer_error(request: Request, error: ResumeAnalyzerError):
    """Handle all custom Resume Analyzer exceptions"""
    logger.warning(f"{error.__class__.__name__}: {error.message}")
    return JSONResponse(
        status_code=error.status_code,
        content=error.to_dict()
    )


@app.exception_handler(413)
async def request_entity_too_large(request: Request, exc):
    """Handle file too large error"""
    return JSONResponse(
        status_code=413,
        content={'error': 'File size exceeds 16MB limit'}
    )


@app.exception_handler(404)
async def not_found(request: Request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={'error': 'Endpoint not found'}
    )


@app.exception_handler(500)
async def internal_error(request: Request, exc):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={'error': 'Internal server error'}
    )
