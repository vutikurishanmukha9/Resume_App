"""
JD-Resume Match Route for AI Resume Analyzer

Handles resume-to-job-description matching with detailed breakdown.
"""

import logging
import traceback
from fastapi import APIRouter, UploadFile, File, Form, Request, HTTPException
from starlette.concurrency import run_in_threadpool

from backend.config import MAX_JD_LENGTH
from backend.services.model_manager import model_manager
from backend.services.analysis import calculate_jd_resume_match
from backend.services.analytics import track_analysis
from backend.rate_limiter import limiter, rate_limiting_enabled
from backend.utils.text_processing import (
    allowed_file,
    read_upload_bytes,
    extract_text_from_bytes,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/match_jd_resume")
async def match_jd_resume(
    request: Request,
    resume: UploadFile = File(...),
    jd_text: str = Form(...)
):
    """Calculate resume-JD match percentage with detailed breakdown"""
    try:
        if not model_manager.is_loaded():
            raise HTTPException(status_code=503, detail='System is still initializing. Please try again.')

        jd_text = jd_text.strip()

        if not jd_text:
            raise HTTPException(status_code=400, detail='Please provide a job description')
        if len(jd_text) > MAX_JD_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f'Job description too long ({len(jd_text)} chars). Maximum is {MAX_JD_LENGTH}.'
            )

        if not resume or not resume.filename:
            raise HTTPException(status_code=400, detail='Please upload a resume file')

        if not allowed_file(resume.filename):
            raise HTTPException(status_code=400, detail='Invalid file type. Only PDF and TXT files are allowed.')

        # Read into memory (no disk I/O)
        file_bytes = await read_upload_bytes(resume)
        resume_text = extract_text_from_bytes(file_bytes, resume.filename)

        # Run CPU-bound analysis in thread pool
        match_percentage, detailed_results = await run_in_threadpool(
            calculate_jd_resume_match, resume_text, jd_text
        )

        track_analysis('jd_match', {
            'match_percentage': match_percentage,
            'component_scores': detailed_results.get('component_scores', {})
        })

        return {
            'success': True,
            'match_percentage': match_percentage,
            'component_scores': detailed_results.get('component_scores', {}),
            'missing_keywords': detailed_results.get('missing_keywords', {}),
            'keyword_suggestions': detailed_results.get('keyword_suggestions', []),
            'skills_breakdown': detailed_results.get('skills_breakdown', {}),
            'message': f"The resume matches {match_percentage}% with the job description"
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
        logger.error(f"JD Match Error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail='Failed to calculate match percentage. Please try again.')
