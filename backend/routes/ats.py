"""
ATS Score Route for AI Resume Analyzer

Handles ATS score calculation with configurable weights and analysis modes.
"""

import logging
import traceback
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Request, HTTPException
from starlette.concurrency import run_in_threadpool
from sentence_transformers import util

from backend.config import MAX_TEXT_LENGTH
from backend.services.model_manager import model_manager
from backend.services.ats_scorer import ATSScorer
from backend.services.analytics import track_analysis
from backend.rate_limiter import limiter, rate_limiting_enabled
from backend.utils.text_processing import (
    allowed_file,
    read_upload_bytes,
    extract_text_from_bytes,
)
from backend.utils.keyword_extractor import extract_keywords
from backend.utils.skill_extractor import extract_skills, get_all_skills_flat

logger = logging.getLogger(__name__)

router = APIRouter()


def _calculate_ats(resume_text, jd_text, jd_title, mode, required_years,
                   required_education, resume_skills, jd_keywords):
    """CPU-bound ATS calculation, run via threadpool."""
    semantic_similarity = None
    if mode == 'deep' and model_manager.embed_model:
        try:
            resume_embedding = model_manager.embed_model.encode(
                resume_text[:MAX_TEXT_LENGTH], convert_to_tensor=True
            )
            jd_embedding = model_manager.embed_model.encode(
                jd_text[:MAX_TEXT_LENGTH], convert_to_tensor=True
            )
            semantic_similarity = float(util.cos_sim(resume_embedding, jd_embedding)[0][0])
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")

    scorer = ATSScorer(mode=mode)
    return scorer.calculate_ats_score(
        resume_text=resume_text,
        jd_text=jd_text,
        jd_title=jd_title,
        required_years=required_years,
        required_education=required_education,
        resume_skills=resume_skills,
        jd_keywords=jd_keywords,
        semantic_similarity=semantic_similarity,
    )


@router.post("/ats_score")
@limiter.limit("10/minute") if rate_limiting_enabled else lambda f: f
async def calculate_ats_score(
    request: Request,
    resume: UploadFile = File(...),
    jd_text: str = Form(...),
    jd_title: str = Form(""),
    mode: str = Form("deep"),
    required_years: int = Form(0),
    required_education: Optional[str] = Form(None),
):
    """Calculate ATS score with detailed breakdown"""
    try:
        if not model_manager.is_loaded():
            raise HTTPException(status_code=503, detail='System is still initializing. Please try again.')

        jd_text = jd_text.strip()
        jd_title = jd_title.strip()
        mode = mode.lower()
        if mode not in ('quick', 'deep'):
            mode = 'deep'

        if not jd_text:
            raise HTTPException(status_code=400, detail='Please provide a job description')

        if not resume or not resume.filename:
            raise HTTPException(status_code=400, detail='Please upload a resume file')

        if not allowed_file(resume.filename):
            raise HTTPException(status_code=400, detail='Invalid file type. Only PDF and TXT files are allowed.')

        # Read into memory (no disk I/O)
        file_bytes = await read_upload_bytes(resume)
        resume_text = extract_text_from_bytes(file_bytes, resume.filename)

        # Extract skills & keywords (fast, can stay in event loop)
        resume_skills_dict = extract_skills(resume_text)
        resume_skills = list(get_all_skills_flat(resume_skills_dict))
        jd_keywords = extract_keywords(jd_text)

        # Run CPU-bound scoring in thread pool
        ats_result = await run_in_threadpool(
            _calculate_ats,
            resume_text, jd_text, jd_title, mode,
            required_years, required_education,
            resume_skills, jd_keywords,
        )

        track_analysis('ats_score', {
            'ats_score': ats_result.get('ats_score', 0),
            'mode': mode,
            'sub_scores': ats_result.get('sub_scores', {})
        })

        return {'success': True, **ats_result}

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Validation error in ATS score: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except (OSError, IOError) as e:
        logger.error(f"File processing error: {e}")
        raise HTTPException(
            status_code=422,
            detail='Could not read the uploaded file. It may be corrupted or in an unsupported format.'
        )
    except Exception as e:
        logger.error(f"ATS Score Error: {e}")
        logger.error(traceback.format_exc())
        error_msg = str(e).lower()
        if 'pdf' in error_msg or 'extract' in error_msg or 'parse' in error_msg:
            detail = 'Failed to parse the resume file. Please ensure it is a valid PDF or TXT file.'
        elif 'encode' in error_msg or 'model' in error_msg or 'embed' in error_msg:
            detail = 'The analysis engine encountered an error. Please try again in a moment.'
        else:
            detail = 'Failed to calculate ATS score. Please try again.'
        raise HTTPException(status_code=500, detail=detail)
