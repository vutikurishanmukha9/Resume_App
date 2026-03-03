"""
JD-Resume Match Route for AI Resume Analyzer

Handles resume-to-job-description matching with detailed breakdown.
"""

import os
import logging
import traceback
from fastapi import APIRouter, UploadFile, File, Form, Request, HTTPException
from werkzeug.utils import secure_filename

from backend.config import UPLOAD_FOLDER
from backend.rate_limiter import limiter, rate_limiting_enabled
from backend.services.model_manager import model_manager
from backend.services.analysis import calculate_jd_resume_match
from backend.services.analytics import track_analysis
from backend.utils.text_processing import (
    allowed_file,
    temporary_file,
    extract_text_from_file
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/match_jd_resume")
@limiter.limit("10/minute") if rate_limiting_enabled else lambda f: f
async def match_jd_resume(
    request: Request,
    resume: UploadFile = File(...),
    jd_text: str = Form(...)
):
    """Calculate resume-JD match percentage with detailed breakdown"""
    try:
        # Check if models are loaded
        if not model_manager.is_loaded():
            raise HTTPException(status_code=503, detail='System is still initializing. Please try again.')

        # Validate inputs
        jd_text = jd_text.strip()
        
        if not jd_text:
            raise HTTPException(status_code=400, detail='Please provide a job description')
        
        if not resume or not resume.filename:
            raise HTTPException(status_code=400, detail='Please upload a resume file')
        
        if not allowed_file(resume.filename):
            raise HTTPException(status_code=400, detail='Invalid file type. Only PDF and TXT files are allowed.')

        # Process file
        filename = secure_filename(resume.filename)
        if not filename:
            raise HTTPException(status_code=400, detail='Invalid filename')
        
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        with temporary_file(file_path):
            # Save uploaded file
            content = await resume.read()
            with open(file_path, 'wb') as f:
                f.write(content)
            
            # Extract text
            resume_text = extract_text_from_file(file_path, filename)
            
            # Calculate match with detailed breakdown
            match_percentage, detailed_results = calculate_jd_resume_match(resume_text, jd_text)
            
            # Track analytics
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
        logger.warning(f"Validation error in JD match: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"JD Match Error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail='Failed to calculate match percentage. Please try again.')
