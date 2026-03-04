import re
import logging
import numpy as np
from typing import Tuple, List, Dict, Any
from sentence_transformers import util

from backend.config import MIN_TEXT_LENGTH, TOP_MATCHES, MATCHING_WEIGHTS, MAX_TEXT_LENGTH
from backend.services.model_manager import model_manager
from backend.exceptions import AnalysisError, TextExtractionError
from backend.utils.keyword_extractor import (
    extract_keywords,
    calculate_keyword_overlap,
    get_missing_keywords,
    split_into_sentences
)
from backend.utils.skill_extractor import (
    extract_skills,
    get_all_skills_flat,
    calculate_skills_match
)
from backend.utils.feature_extractor import extract_resume_features

logger = logging.getLogger(__name__)


# ── Salary adjustment by job role ─────────────────────────────────
# The salary model was trained on generic features.  This table
# applies a role-specific multiplier so that a Software Engineer
# and a Retail Manager with the same features get different salaries.
JOB_SALARY_MULTIPLIER = {
    'data science':    1.15,
    'software':        1.10,
    'web development': 1.05,
    'devops':          1.12,
    'cloud':           1.12,
    'cybersecurity':   1.10,
    'database':        1.05,
    'network':         1.00,
    'project manager': 1.00,
    'business analyst':1.00,
    'hr':              0.85,
    'sales':           0.90,
    'marketing':       0.90,
    'finance':         1.05,
    'healthcare':      0.95,
    'education':       0.80,
    'design':          0.95,
    'mechanical':      1.00,
    'electrical':      1.05,
    'civil':           0.95,
}


def _get_salary_multiplier(job_title: str) -> float:
    """Return a salary multiplier for the predicted job category."""
    title_lower = job_title.lower()
    for key, mult in JOB_SALARY_MULTIPLIER.items():
        if key in title_lower:
            return mult
    return 1.0  # default: no adjustment


# ── Contact / header line detector ────────────────────────────────
_CONTACT_PATTERNS = re.compile(
    r'(?:'
    r'[\w.+-]+@[\w-]+\.[\w.]+|'         # email
    r'\+?\d[\d\s\-()]{7,}|'              # phone
    r'linkedin\.com|github\.com|'        # profile URLs
    r'\b(?:address|phone|email|mobile|contact)\b'
    r')',
    re.IGNORECASE
)

_HEADER_PHRASES = {
    'about the company', 'about us', 'who we are',
    'equal opportunity', 'we are committed', 'our mission',
    'company overview', 'job summary', 'position overview',
}


def _is_substantive_sentence(sentence: str) -> bool:
    """Return True if the sentence contains real content, not contact info or headers."""
    s = sentence.strip()
    if len(s) < 30:  # too short to be meaningful
        return False
    if _CONTACT_PATTERNS.search(s):
        return False
    s_lower = s.lower()
    for phrase in _HEADER_PHRASES:
        if phrase in s_lower:
            return False
    return True


def analyze_resume(resume_text: str) -> Tuple[str, List[Tuple[str, float]], float, Dict[str, Any]]:
    """
    Predict job, matches, and salary with error handling.
    Returns: (predicted_job, matches, predicted_salary, salary_details)
    """
    try:
        # Validate input
        if not resume_text or len(resume_text.strip()) < MIN_TEXT_LENGTH:
            raise ValueError("Resume text is too short for analysis")
        
        # Predict job category
        predicted_job = model_manager.resume_classifier.predict([resume_text])[0]

        # Generate resume embedding
        resume_embed = model_manager.embed_model.encode(
            [resume_text], 
            convert_to_tensor=True
        )
        
        # Calculate cosine similarities
        cosine_scores = util.cos_sim(resume_embed, model_manager.job_embeddings)
        top_indices = np.argsort(-cosine_scores[0].cpu().numpy())[:TOP_MATCHES]

        # Get top matches
        matches = []
        for idx in top_indices:
            try:
                job_title = model_manager.job_df.iloc[idx]['Job Title']
                score = float(cosine_scores[0][idx])
                matches.append((job_title, score))
            except (IndexError, KeyError) as e:
                logger.warning(f"Failed to process match at index {idx}: {e}")
                continue

        # Extract features for salary prediction
        features = extract_resume_features(resume_text)
        
        # Create feature vector for salary model
        # Pass raw features — the model learns the weights, not us
        feature_vector = np.array([[
            features['years_experience'],
            features['education_level'],
            features['seniority_level'],
            features['skills_count'],
        ]])
        
        # Predict base salary, then adjust by job role
        base_salary = float(model_manager.salary_model.predict(feature_vector)[0])
        role_multiplier = _get_salary_multiplier(predicted_job)
        predicted_salary = base_salary * role_multiplier
        
        # Prepare salary details for response
        salary_details = {
            'predicted_job': predicted_job,
            'role_multiplier': role_multiplier,
            'features': {
                'years_experience': features['years_experience'],
                'education_level': features['education_level'],
                'seniority_level': features['seniority_level'],
                'skills_count': features['skills_count']
            },
            'confidence': round(features['completeness'], 2),
            'note': 'Salary adjusted for predicted job role. Confidence indicates feature completeness.'
        }
        
        return predicted_job, matches, predicted_salary, salary_details
        
    except ValueError:
        raise
    except (KeyError, IndexError, RuntimeError) as e:
        logger.error(f"Resume analysis failed: {e}")
        raise AnalysisError(f"Failed to analyze resume: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in resume analysis: {e}")
        raise AnalysisError(f"Failed to analyze resume: {str(e)}")


def calculate_jd_resume_match(resume_text: str, jd_text: str) -> Tuple[float, Dict[str, Any]]:
    """
    Enhanced JD-Resume matching with multiple scoring components.
    Returns: (final_score, detailed_results)
    """
    try:
        # Validate inputs
        resume_text = resume_text.strip()
        jd_text = jd_text.strip()
        
        if not resume_text or len(resume_text) < MIN_TEXT_LENGTH:
            raise ValueError("Resume text is too short for matching")
        
        if not jd_text or len(jd_text) < 20:
            raise ValueError("Job description is too short for matching")
        
        component_scores = {}
        detailed_results = {}
        
        # 1. SEMANTIC SIMILARITY (40% weight)
        resume_embedding = model_manager.embed_model.encode(
            resume_text, 
            convert_to_tensor=True
        )
        jd_embedding = model_manager.embed_model.encode(
            jd_text, 
            convert_to_tensor=True
        )
        raw_semantic = util.cos_sim(resume_embedding, jd_embedding).item()
        
        # Calibrate semantic score.
        # Raw cosine similarity for all-MiniLM-L6-v2 typically ranges:
        #   0.0–0.2  = unrelated texts
        #   0.2–0.3  = loosely related
        #   0.3–0.5  = moderately related
        #   0.5–0.7  = closely related
        #   0.7+     = near-identical meaning
        # We map this to a 0–100 user-facing score.
        if raw_semantic < 0.2:
            semantic_score = raw_semantic * 100           # 0–20 → 0–20
        elif raw_semantic < 0.3:
            semantic_score = 30 + (raw_semantic - 0.2) * 200   # 0.2–0.3 → 30–50
        else:
            semantic_score = min(98, 50 + (raw_semantic - 0.3) * 112.5)  # 0.3+ → 50–98
        
        component_scores['semantic'] = semantic_score
        
        # 2. KEYWORD OVERLAP (30% weight)
        resume_keywords = extract_keywords(resume_text)
        jd_keywords = extract_keywords(jd_text)
        keyword_score = calculate_keyword_overlap(resume_keywords, jd_keywords)
        component_scores['keyword'] = keyword_score
        
        # Get missing keywords with importance ranking
        missing_keywords = get_missing_keywords(resume_keywords, jd_keywords)
        detailed_results['missing_keywords'] = missing_keywords
        
        # Generate keyword suggestions
        suggestions = []
        if missing_keywords['critical']:
            suggestions.append(f"Add critical keywords: {', '.join(missing_keywords['critical'][:5])}")
        if missing_keywords['important']:
            suggestions.append(f"Consider adding: {', '.join(missing_keywords['important'][:5])}")
        detailed_results['keyword_suggestions'] = suggestions
        
        # 3. SKILLS MATCHING (20% weight)
        resume_skills_categorized = extract_skills(resume_text)
        jd_skills_categorized = extract_skills(jd_text)
        
        # Flatten for overall matching
        resume_skills_flat = get_all_skills_flat(resume_skills_categorized)
        jd_skills_flat = get_all_skills_flat(jd_skills_categorized)
        
        skills_score = calculate_skills_match(resume_skills_flat, jd_skills_flat)
        component_scores['skills'] = skills_score
        
        # Detailed skills breakdown
        missing_skills_by_category = {}
        matched_skills_by_category = {}
        
        for category in resume_skills_categorized.keys():
            resume_cat_skills = resume_skills_categorized.get(category, set())
            jd_cat_skills = jd_skills_categorized.get(category, set())
            
            if jd_cat_skills:
                missing = jd_cat_skills - resume_cat_skills
                matched = resume_cat_skills & jd_cat_skills
                
                if missing:
                    missing_skills_by_category[category] = list(missing)
                if matched:
                    matched_skills_by_category[category] = list(matched)
        
        detailed_results['skills_breakdown'] = {
            'resume_skills': {k: list(v) for k, v in resume_skills_categorized.items() if v},
            'missing_skills': missing_skills_by_category,
            'matched_skills': matched_skills_by_category
        }
        
        # 4. CONTEXTUAL SIMILARITY (10% weight)
        # Filter out contact info, headers, and very short lines
        # so we compare actual experience/skills content, not
        # "John Doe" vs "About Our Company".
        resume_sentences = [
            s for s in split_into_sentences(resume_text)
            if _is_substantive_sentence(s)
        ][:12]
        jd_sentences = [
            s for s in split_into_sentences(jd_text)
            if _is_substantive_sentence(s)
        ][:12]
        
        if resume_sentences and jd_sentences:
            resume_sent_embeds = model_manager.embed_model.encode(
                resume_sentences, 
                convert_to_tensor=True
            )
            jd_sent_embeds = model_manager.embed_model.encode(
                jd_sentences, 
                convert_to_tensor=True
            )
            
            similarities = util.cos_sim(jd_sent_embeds, resume_sent_embeds)
            max_sims = similarities.max(dim=1).values
            contextual_score = max_sims.mean().item() * 100
        else:
            contextual_score = semantic_score
        
        component_scores['context'] = contextual_score
        
        # Calculate weighted final score
        final_score = (
            component_scores['semantic'] * MATCHING_WEIGHTS['semantic'] +
            component_scores['keyword'] * MATCHING_WEIGHTS['keyword'] +
            component_scores['skills'] * MATCHING_WEIGHTS['skills'] +
            component_scores['context'] * MATCHING_WEIGHTS['context']
        )
        
        # Ensure score is between 0 and 100
        final_score = round(max(0, min(100, final_score)), 2)
        
        # Round component scores
        for key in component_scores:
            component_scores[key] = round(component_scores[key], 2)
        
        # Add component scores to detailed results
        detailed_results['component_scores'] = component_scores
        
        logger.info(f"Match scores - Final: {final_score}%, Components: {component_scores}")
        
        return final_score, detailed_results
        
    except ValueError:
        raise
    except (KeyError, IndexError, RuntimeError) as e:
        logger.error(f"JD matching failed: {e}")
        raise AnalysisError(f"Failed to calculate match: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in JD matching: {e}")
        raise AnalysisError(f"Failed to calculate match: {str(e)}")
