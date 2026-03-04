"""
Upload Route for AI Resume Analyzer

Handles resume file upload and analysis (job prediction, salary, matching).
"""

import os
import logging
import traceback
from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from werkzeug.utils import secure_filename

from backend.config import UPLOAD_FOLDER
from backend.rate_limiter import limiter, rate_limiting_enabled
from backend.services.model_manager import model_manager
from backend.services.analysis import analyze_resume
from backend.services.analytics import track_analysis
from backend.utils.text_processing import (
    allowed_file,
    temporary_file,
    extract_text_from_file,
    save_upload_safely
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/upload")
@limiter.limit("10/minute") if rate_limiting_enabled else lambda f: f
async def upload_resume(request: Request, resume: UploadFile = File(...)):
    """Handle resume upload and analysis"""
    try:
        # Check if models are loaded
        if not model_manager.is_loaded():
            raise HTTPException(status_code=503, detail='System is still initializing. Please try again.')

        # Validate file
        if not resume or not resume.filename:
            raise HTTPException(status_code=400, detail='No file selected')
        
        if not allowed_file(resume.filename):
            raise HTTPException(status_code=400, detail='Invalid file type. Only PDF and TXT files are allowed.')

        # Secure filename
        filename = secure_filename(resume.filename)
        if not filename:
            raise HTTPException(status_code=400, detail='Invalid filename')
        
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        # Process file
        with temporary_file(file_path):
            # Save uploaded file in chunks with size limit
            await save_upload_safely(resume, file_path)
            
            # Extract text
            resume_text = extract_text_from_file(file_path, filename)
            
            # Analyze resume
            predicted_job, matches, salary, salary_details = analyze_resume(resume_text)
            
            # Track analytics
            track_analysis('upload', {
                'predicted_job': predicted_job,
                'salary': int(salary),
                'confidence': salary_details['confidence']
            })

            return {
                'success': True,
                'predicted_job': predicted_job,
                'matches': [{'title': t, 'score': f"{s:.3f}"} for t, s in matches],
                'salary': f"₹{int(salary):,}",
                'salary_details': salary_details
            }

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except (OSError, IOError) as e:
        logger.error(f"File processing error: {e}")
        raise HTTPException(
            status_code=422,
            detail='Could not read the uploaded file. It may be corrupted, password-protected, or in an unsupported format.'
        )
    except Exception as e:
        logger.error(f"Upload error: {e}")
        logger.error(traceback.format_exc())
        error_msg = str(e).lower()
        if 'pdf' in error_msg or 'extract' in error_msg or 'parse' in error_msg:
            detail = 'Failed to parse the resume file. Please ensure it is a valid, non-corrupted PDF or TXT file.'
        elif 'encode' in error_msg or 'model' in error_msg or 'embed' in error_msg:
            detail = 'The analysis engine encountered an error. Please try again in a moment.'
        else:
            detail = 'An unexpected error occurred. Please try again.'
        raise HTTPException(status_code=500, detail=detail)
