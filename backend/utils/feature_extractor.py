"""
Feature Extraction Utilities for AI Resume Analyzer

Extracts experience years, education level, seniority level,
and combined resume features for salary prediction.
"""

import re
from typing import Dict, Any
from datetime import datetime

from backend.utils.skill_extractor import extract_skills, get_all_skills_flat


def _parse_month(month_str: str) -> int:
    """Convert a month abbreviation to its 1-based number."""
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    }
    return month_map.get(month_str[:3].lower(), 1)


def _extract_experience_section(text: str) -> str:
    """
    Try to isolate the EXPERIENCE / WORK HISTORY section from resume text.
    Returns just that section, or the full text if no section header is found.
    Works with messy PDF-extracted text where newlines aren't always clean.
    """
    # Common section headers for experience
    exp_headers = [
        r'(?:professional\s+)?experience',
        r'work\s+(?:experience|history)',
        r'employment\s+history',
        r'career\s+(?:history|summary)',
        r'internship(?:s)?',
    ]
    # Section headers that typically follow experience
    next_headers = [
        r'education', r'skills', r'certifications?', r'projects?',
        r'awards?', r'publications?', r'interests?', r'hobbies',
        r'references?', r'activities', r'achievements?', r'summary',
        r'objective', r'languages?', r'volunteer',
    ]

    text_lower = text.lower()

    # Try multiple matching strategies (strict → relaxed)
    patterns = [
        r'(?:^|\n)\s*(?:' + '|'.join(exp_headers) + r')\s*\n',
        r'(?:^|\n)\s*(?:' + '|'.join(exp_headers) + r')\s*(?:\n|.)',
        r'\b(?:' + '|'.join(exp_headers) + r')\b',
    ]

    exp_match = None
    for pat in patterns:
        exp_match = re.search(pat, text_lower)
        if exp_match:
            break

    if exp_match:
        start = exp_match.end()

        # Find the next section header after experience
        next_patterns = [
            r'(?:^|\n)\s*(?:' + '|'.join(next_headers) + r')\s*(?:\n|$)',
            r'\b(?:' + '|'.join(next_headers) + r')\s*(?:\n|$)',
        ]
        next_match = None
        for np in next_patterns:
            next_match = re.search(np, text_lower[start:])
            if next_match:
                break

        end = start + next_match.start() if next_match else len(text)
        return text[start:end]

    return text  # Fallback: use full text


def extract_years_of_experience(text: str) -> float:
    """
    Extract years of experience from resume text.

    Strategy (in priority order):
    1. Explicit statements like "X years of experience".
    2. Month-level date-range parsing (e.g. "Jun 2024 – Aug 2024")
       from the EXPERIENCE section only.
    3. Year-only date-range fallback ("2019 – 2023").
    """
    text_lower = text.lower()
    now = datetime.now()
    current_year = now.year
    current_month = now.month

    # ── 1. Explicit text patterns ──────────────────────────────────
    explicit_years: list[float] = []

    # "X years of experience" / "X+ years"
    for m in re.finditer(r'(\d+)\+?\s*years?\s*(?:of)?\s*(?:experience|exp|expertise)', text_lower):
        explicit_years.append(float(m.group(1)))

    # "X-Y years of experience" → take the higher number
    for m in re.finditer(r'(\d+)\s*[-–]\s*(\d+)\s*years?\s*(?:of)?\s*(?:experience|exp)', text_lower):
        explicit_years.append(float(m.group(2)))

    if explicit_years:
        return max(explicit_years)

    # ── 2. Month+Year date ranges (from EXPERIENCE section) ────────
    # Use only the experience section to avoid counting education dates
    exp_text = _extract_experience_section(text).lower()

    month_range_pattern = (
        r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*'
        r'((?:19|20)\d{2})\s*'
        r'[-–—]+\s*'
        r'(?:(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*'
        r'((?:19|20)\d{2})|'
        r'(present|current|till\s*date|to\s*date|ongoing))'
    )
    total_months = 0
    range_matches = list(re.finditer(month_range_pattern, exp_text))

    if range_matches:
        for m in range_matches:
            start_month = _parse_month(m.group(1))
            start_year = int(m.group(2))

            if m.group(5):  # "present" / "current"
                end_month = current_month
                end_year = current_year
            else:
                end_month = _parse_month(m.group(3))
                end_year = int(m.group(4))

            # Calculate duration in months
            duration = (end_year - start_year) * 12 + (end_month - start_month)
            if duration > 0:
                total_months += duration

        if total_months > 0:
            return round(total_months / 12, 1)

    # ── 3. Year-only fallback ──────────────────────────────────────
    year_pattern = r'\b((?:19|20)\d{2})\b'
    year_strs = re.findall(year_pattern, exp_text)
    if re.search(r'\b(present|current|till\s*date|to\s*date|ongoing)\b', exp_text):
        year_strs.append(str(current_year))

    if len(year_strs) >= 2:
        unique_years = sorted(set(int(y) for y in year_strs))
        span = min(unique_years[-1], current_year) - unique_years[0]
        if 0 < span < 50:
            return float(span)

    return 0.0


def extract_education_level(text: str) -> int:
    """
    Extract education level from resume.
    Returns: 0=Unknown, 1=Diploma, 2=Bachelor's, 3=Master's, 4=PhD
    
    Delegates to ats_helpers.detect_education_level() for single source of truth.
    """
    from backend.services.ats_helpers import detect_education_level
    result = detect_education_level(text)
    return result.get('level_score', 0)


def extract_seniority_level(text: str) -> int:
    """
    Extract job seniority level from resume.
    Returns: 0=Entry, 1=Mid, 2=Senior, 3=Lead/Principal
    
    Delegates to ats_helpers.detect_seniority_level() for single source of truth.
    """
    from backend.services.ats_helpers import detect_seniority_level
    result = detect_seniority_level(text)
    level = result.get('level', 'unknown')
    # Map level name to int
    level_map = {'entry': 0, 'mid': 1, 'senior': 2, 'lead': 3}
    return level_map.get(level, 0)  # Default entry-level, not mid (#17)


def extract_resume_features(resume_text: str) -> Dict[str, Any]:
    """
    Extract all features from resume for salary prediction.
    Returns a dictionary with features and their values.
    """
    features = {}
    
    # Extract years of experience
    features['years_experience'] = extract_years_of_experience(resume_text)
    
    # Extract education level
    features['education_level'] = extract_education_level(resume_text)
    
    # Extract seniority level
    features['seniority_level'] = extract_seniority_level(resume_text)
    
    # Extract skills count
    skills_categorized = extract_skills(resume_text)
    skills_flat = get_all_skills_flat(skills_categorized)
    features['skills_count'] = len(skills_flat)
    
    # Calculate feature completeness for confidence score
    completeness_factors = [
        features['years_experience'] > 0,
        features['education_level'] > 0,
        features['seniority_level'] > 0,
        features['skills_count'] > 0
    ]
    features['completeness'] = sum(completeness_factors) / len(completeness_factors)
    
    return features

