import logging
import traceback as tb_mod
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Request, HTTPException
from starlette.concurrency import run_in_threadpool

from backend.config import CURRENCY_SYMBOL, MAX_JD_LENGTH
from backend.services.model_manager import model_manager
from backend.services.analysis import analyze_resume, calculate_jd_resume_match
from backend.services.ats_scorer import ATSScorer
from backend.services.analytics import track_analysis
from backend.utils.text_processing import (
    allowed_file,
    read_upload_bytes,
    extract_text_from_bytes,
)
from backend.utils.keyword_extractor import extract_keywords
from backend.utils.skill_extractor import extract_skills, get_all_skills_flat

logger = logging.getLogger(__name__)

router = APIRouter()


def _run_full_analysis(
    resume_text: str,
    jd_text: str,
    jd_title: str,
    mode: str,
    required_years: int,
    required_education: Optional[str],
) -> dict:
    """
    CPU-bound work: run all analysis in one shot.

    This function is called via run_in_threadpool so it doesn't
    block the async event loop.
    """
    results = {}

    # ── 1. Upload analysis (job prediction + salary) ──────────
    try:
        predicted_job, matches, salary, salary_details = analyze_resume(resume_text)
        results['upload'] = {
            'success': True,
            'predicted_job': predicted_job,
            'matches': [{'title': t, 'score': f"{s:.3f}"} for t, s in matches],
            'salary': f"{CURRENCY_SYMBOL}{int(salary):,}",
            'salary_details': salary_details,
        }
    except Exception as e:
        logger.warning(f"Upload analysis failed: {e}")
        logger.debug(tb_mod.format_exc())
        results['upload'] = {'success': False, 'error': str(e)}

    # ── 2. JD Match (semantic + keyword + skills) ─────────────
    try:
        match_pct, detailed = calculate_jd_resume_match(resume_text, jd_text)
        results['jd_match'] = {
            'success': True,
            'match_percentage': match_pct,
            'component_scores': detailed.get('component_scores', {}),
            'missing_keywords': detailed.get('missing_keywords', {}),
            'keyword_suggestions': detailed.get('keyword_suggestions', []),
            'skills_breakdown': detailed.get('skills_breakdown', {}),
        }
    except Exception as e:
        logger.warning(f"JD match failed: {e}")
        logger.debug(tb_mod.format_exc())
        results['jd_match'] = {'success': False, 'error': str(e)}

    # ── 3. ATS Score ──────────────────────────────────────────
    try:
        from sentence_transformers import util

        resume_skills_dict = extract_skills(resume_text)
        resume_skills = list(get_all_skills_flat(resume_skills_dict))
        jd_keywords = extract_keywords(jd_text)

        semantic_similarity = None
        if mode == 'deep' and model_manager.embed_model:
            try:
                from backend.config import MAX_TEXT_LENGTH
                resume_emb = model_manager.embed_model.encode(
                    resume_text[:MAX_TEXT_LENGTH], convert_to_tensor=True
                )
                jd_emb = model_manager.embed_model.encode(
                    jd_text[:MAX_TEXT_LENGTH], convert_to_tensor=True
                )
                semantic_similarity = float(util.cos_sim(resume_emb, jd_emb)[0][0])
            except Exception as e:
                logger.warning(f"Semantic similarity failed: {e}")

        scorer = ATSScorer(mode=mode)
        ats_result = scorer.calculate_ats_score(
            resume_text=resume_text,
            jd_text=jd_text,
            jd_title=jd_title,
            required_years=required_years,
            required_education=required_education,
            resume_skills=resume_skills,
            jd_keywords=jd_keywords,
            semantic_similarity=semantic_similarity,
        )
        results['ats'] = {'success': True, **ats_result}
    except Exception as e:
        logger.warning(f"ATS scoring failed: {e}")
        logger.debug(tb_mod.format_exc())
        results['ats'] = {'success': False, 'error': str(e)}

    return results


@router.post("/analyze-full")
async def analyze_full(
    request: Request,
    resume: UploadFile = File(...),
    jd_text: str = Form(...),
    jd_title: str = Form(""),
    mode: str = Form("deep"),
    required_years: int = Form(0),
    required_education: Optional[str] = Form(None),
):
    """
    Unified analysis endpoint.

    Parses the resume ONCE, then runs upload analysis, JD match,
    and ATS scoring all in memory. Returns a combined response.
    """
    try:
        if not model_manager.is_loaded():
            raise HTTPException(
                status_code=503,
                detail='System is still initializing. Please try again.',
            )

        jd_text = jd_text.strip()
        jd_title = jd_title.strip()
        mode = mode.lower()
        if mode not in ('quick', 'deep'):
            mode = 'deep'

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
            raise HTTPException(
                status_code=400,
                detail='Invalid file type. Only PDF and TXT files are allowed.',
            )

        # ── Read file into memory (no disk I/O) ──────────────
        file_bytes = await read_upload_bytes(resume)
        resume_text = extract_text_from_bytes(file_bytes, resume.filename)

        # ── Run all CPU-bound analysis in a thread pool ──────
        results = await run_in_threadpool(
            _run_full_analysis,
            resume_text,
            jd_text,
            jd_title,
            mode,
            required_years,
            required_education,
        )

        # Track analytics
        track_analysis('analyze_full', {
            'ats_score': results.get('ats', {}).get('ats_score', 0),
            'match_pct': results.get('jd_match', {}).get('match_percentage', 0),
            'mode': mode,
        })

        return {'success': True, **results}

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (OSError, IOError) as e:
        logger.error(f"File processing error: {e}")
        raise HTTPException(
            status_code=422,
            detail='Could not read the uploaded file. It may be corrupted or in an unsupported format.',
        )
    except Exception as e:
        logger.error(f"Analyze-full error: {e}")
        logger.error(tb_mod.format_exc())
        raise HTTPException(status_code=500, detail='Analysis failed. Please try again.')
