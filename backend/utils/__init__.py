"""
Utils package for AI Resume Analyzer

Provides text processing, keyword extraction, skill extraction,
and feature extraction utilities.
"""

from backend.utils.text_processing import (
    allowed_file,
    read_upload_bytes,
    extract_text_from_bytes,
)
from backend.utils.keyword_extractor import (
    extract_keywords,
    calculate_tfidf_weights,
    calculate_keyword_overlap,
    split_into_sentences,
    get_missing_keywords
)
from backend.utils.skill_extractor import (
    extract_skills,
    get_all_skills_flat,
    calculate_skills_match
)
from backend.utils.feature_extractor import (
    extract_years_of_experience,
    extract_education_level,
    extract_seniority_level,
    extract_resume_features
)
