"""
Upload Route for AI Resume Analyzer

Handles resume file upload and analysis (job prediction, salary, matching).
"""

import logging
import traceback
from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from starlette.concurrency import run_in_threadpool

from backend.services.model_manager import model_manager
from backend.services.analysis import analyze_resume
from backend.services.analytics import track_analysis
from backend.rate_limiter import limiter, rate_limiting_enabled
from backend.utils.text_processing import (
    allowed_file,
    read_upload_bytes,
    extract_text_from_bytes,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/upload")
@limiter.limit("10/minute") if rate_limiting_enabled else lambda f: f
async def upload_resume(request: Request, resume: UploadFile = File(...)):
    """Handle resume upload and analysis"""
    try:
        if not model_manager.is_loaded():
            raise HTTPException(status_code=503, detail='System is still initializing. Please try again.')

        if not resume or not resume.filename:
            raise HTTPException(status_code=400, detail='No file selected')

        if not allowed_file(resume.filename):
            raise HTTPException(status_code=400, detail='Invalid file type. Only PDF and TXT files are allowed.')

        # Read into memory (no disk I/O)
        file_bytes = await read_upload_bytes(resume)
        resume_text = extract_text_from_bytes(file_bytes, resume.filename)

        # Run CPU-bound analysis in thread pool
        predicted_job, matches, salary, salary_details = await run_in_threadpool(
            analyze_resume, resume_text
        )

        track_analysis('upload', {
            'predicted_job': predicted_job,
            'salary': int(salary),
            'confidence': salary_details['confidence']
        })

        return {
            'success': True,
            'predicted_job': predicted_job,
            'matches': [{'title': t, 'score': f"{s:.3f}"} for t, s in matches],
            'salary': f"\u20b9{int(salary):,}",
            'salary_details': salary_details
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (OSError, IOError) as e:
        logger.error(f"File processing error: {e}")
        raise HTTPException(
            status_code=422,
            detail='Could not read the uploaded file. It may be corrupted or in an unsupported format.'
        )
    except Exception as e:
        logger.error(f"Upload error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail='An unexpected error occurred. Please try again.')
