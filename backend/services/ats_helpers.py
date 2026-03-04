"""
ATS Scorer Helper Functions

Detection, extraction, and scoring utility functions used by the ATSScorer class.
Separated from the main scorer for modularity and testability.
"""

import re
from typing import Dict, List, Any
from collections import Counter
from datetime import datetime

from backend.services.ats_constants import (
    SECTION_HEADERS,
    SECTION_PENALTIES,
    SECTION_KEYWORD_WEIGHT,
    EDUCATION_LEVELS,
    EDUCATION_FALSE_POSITIVES,
    SENIORITY_PATTERNS,
    TITLE_ONTOLOGY,
    ACTION_VERBS,
    REQUIRED_INDICATORS,
    PREFERRED_INDICATORS,
    NEGATION_PHRASES,
    MAX_KEYWORD_REPEATS,
    MAX_KEYWORD_DENSITY,
    STUFFING_PENALTY,
)


# ==================== TEXT UTILITIES ====================

def normalize_text(text: str) -> str:
    """Normalize text for matching: lowercase, remove extra whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_years_from_text(text: str) -> List[int]:
    """Extract year numbers (e.g., 2020, 2023) from text."""
    years = re.findall(r'\b(19|20)\d{2}\b', text)
    return [int(y) for y in years]


# ==================== EXPERIENCE ====================

def extract_experience_duration(text: str) -> Dict[str, Any]:
    """
    Extract experience duration from resume text.

    Delegates to feature_extractor's robust implementation which
    handles section isolation, month-level parsing, and explicit
    experience statements, then adapts the result into the dict
    format the ATS scorer expects.
    """
    from backend.utils.feature_extractor import extract_years_of_experience

    total_years = extract_years_of_experience(text)

    result = {
        'total_years': total_years,
        'skill_years': {},
        'date_ranges': []
    }

    # Supplement with skill-specific experience patterns
    skill_year_patterns = [
        r'(\w+(?:\.\w+)?)\s*[\(\[]?\s*(\d+)\+?\s*(?:years?|yrs?)[\)\]]?',
        r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(\w+(?:\.\w+)?)',
    ]

    for pattern in skill_year_patterns:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            if len(match) == 2:
                if match[0].isdigit():
                    years, skill = int(match[0]), match[1]
                else:
                    skill, years = match[0], int(match[1])
                if len(skill) > 1 and years <= 50:  # sanity check
                    result['skill_years'][skill] = years

    return result


# ==================== EDUCATION ====================

def detect_education_level(text: str) -> Dict[str, Any]:
    """Detect highest education level from resume text.

    Uses phrase-level matching to avoid false positives like
    'Scrum Master' or 'Mastered Python' being detected as Master's degree.
    """
    text_lower = normalize_text(text)

    result = {
        'highest_level': 'unknown',
        'level_score': 0,
        'degrees_found': []
    }

    # Check for false positives first
    has_false_positive = {}
    for fp in EDUCATION_FALSE_POSITIVES:
        if fp in text_lower:
            has_false_positive[fp] = True

    level_scores = {'phd': 4, 'masters': 3, 'bachelors': 2, 'diploma': 1}

    for level, keywords in EDUCATION_LEVELS.items():
        for keyword in keywords:
            if keyword in text_lower:
                # Verify it's not a false positive
                is_false = False
                for fp in has_false_positive:
                    if keyword.rstrip("'s").rstrip(' of').rstrip(' in') in fp:
                        is_false = True
                        break

                if not is_false:
                    result['degrees_found'].append(level)
                    if level_scores.get(level, 0) > result['level_score']:
                        result['highest_level'] = level
                        result['level_score'] = level_scores[level]
                    break

    return result


# ==================== SENIORITY ====================

def detect_seniority_level(text: str) -> Dict[str, Any]:
    """Detect seniority level from resume text."""
    text_lower = normalize_text(text)

    result = {
        'level': 'unknown',
        'level_score': 0,
        'indicators_found': []
    }

    level_scores = {'lead': 4, 'senior': 3, 'mid': 2, 'entry': 1}

    for level, keywords in SENIORITY_PATTERNS.items():
        for keyword in keywords:
            if keyword in text_lower:
                result['indicators_found'].append(keyword)
                if level_scores.get(level, 0) > result['level_score']:
                    result['level'] = level
                    result['level_score'] = level_scores[level]

    return result


# ==================== SECTION DETECTION ====================

def detect_resume_sections(text: str) -> Dict[str, str]:
    """
    Detect and extract sections from resume text.
    Returns dictionary with section name -> section content.
    """
    sections = {}
    text_lower = text.lower()

    # Find all potential section headers and their positions
    header_positions = []

    for section_type, headers in SECTION_HEADERS.items():
        for header in headers:
            pattern = rf'(?:^|\n)\s*({re.escape(header)})\s*[:\n]'
            matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
            for match in matches:
                header_positions.append({
                    'type': section_type,
                    'header': header,
                    'start': match.end(),
                    'header_start': match.start()
                })

    # Sort by position
    header_positions.sort(key=lambda x: x['start'])

    # Extract content between headers
    for i, hp in enumerate(header_positions):
        start = hp['start']
        if i + 1 < len(header_positions):
            end = header_positions[i + 1]['header_start']
        else:
            end = len(text)

        content = text[start:end].strip()
        if content and hp['type'] not in sections:
            sections[hp['type']] = content

    return sections


# ==================== ACHIEVEMENTS ====================

def extract_achievements(text: str) -> List[Dict[str, Any]]:
    """
    Extract quantified achievements from text.

    Tightened rules:
    - Action verb must START the sentence/clause (not just appear anywhere)
    - Metric must be within 80 chars of the verb (same clause)
    - Filters out generic number mentions like "team of 5"
    """
    achievements = []

    sentences = re.split(r'[.•\n]', text)

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15:
            continue

        sentence_lower = sentence.lower().lstrip('- ')

        # Action verb must START the sentence/clause
        leading_verb = None
        for verb in ACTION_VERBS:
            if sentence_lower.startswith(verb):
                leading_verb = verb
                break

        if not leading_verb:
            continue

        # Check for metrics NEAR the verb (within the same clause)
        clause = sentence_lower[:120]

        metrics = {
            'percentage': bool(re.search(r'\d+\s*%', clause)),
            'currency': bool(re.search(r'[\$£€₹]\s*\d+|\d+\s*(?:k|m|million|thousand|lakh|crore)', clause)),
            'number': bool(re.search(
                r'\b\d+\+?\s*(?:users?|customers?|clients?|applications?|apis?|'
                r'endpoints?|requests?|deployments?|projects?|microservices?|modules?)\b',
                clause
            )),
            'time': bool(re.search(
                r'\d+\s*(?:hours?|days?|weeks?|months?|x\s*faster|x\s*improvement)',
                clause
            )),
            'multiplier': bool(re.search(r'\b\d+x\b', clause)),
        }

        has_metric = any(metrics.values())

        if has_metric:
            achievements.append({
                'text': sentence,
                'metrics': [k for k, v in metrics.items() if v]
            })

    return achievements


# ==================== KEYWORD CLASSIFICATION ====================

def classify_jd_keywords(jd_text: str, all_keywords: List[str]) -> Dict[str, List[str]]:
    """
    Classify JD keywords as required, preferred, or standard.

    Includes negation detection: "not required" won't flag as required.
    """
    jd_lower = jd_text.lower()

    result = {
        'required': [],
        'preferred': [],
        'standard': []
    }

    sentences = re.split(r'[.•\n]', jd_lower)

    for keyword in all_keywords:
        keyword_lower = keyword.lower()
        classified = False

        for sentence in sentences:
            if keyword_lower in sentence:
                has_negation = any(neg in sentence for neg in NEGATION_PHRASES)

                if has_negation:
                    if keyword not in result['preferred']:
                        result['preferred'].append(keyword)
                    classified = True
                    break
                elif any(ind in sentence for ind in REQUIRED_INDICATORS):
                    if keyword not in result['required']:
                        result['required'].append(keyword)
                    classified = True
                    break
                elif any(ind in sentence for ind in PREFERRED_INDICATORS):
                    if keyword not in result['preferred']:
                        result['preferred'].append(keyword)
                    classified = True
                    break

        if not classified and keyword not in result['standard']:
            result['standard'].append(keyword)

    return result


# ==================== JOB TITLE MATCHING ====================

def match_job_title(resume_text: str, jd_title: str) -> Dict[str, Any]:
    """
    Match resume job titles against JD title using ontology.

    Falls back gracefully for non-tech roles: if the JD title doesn't
    match any ontology category, uses direct text matching with a
    partial score instead of silently returning 0.
    """
    resume_lower = normalize_text(resume_text)
    jd_lower = normalize_text(jd_title)

    result = {
        'score': 0,
        'matched_category': None,
        'resume_titles': [],
        'jd_category': None
    }

    # Find JD title category
    for category, titles in TITLE_ONTOLOGY.items():
        if any(title in jd_lower for title in titles):
            result['jd_category'] = category
            break

    # Find resume title matches
    for category, titles in TITLE_ONTOLOGY.items():
        for title in titles:
            if title in resume_lower:
                result['resume_titles'].append(title)
                if category == result['jd_category']:
                    result['matched_category'] = category
                    result['score'] = 100
                elif result['score'] < 60:
                    result['score'] = 60

    # Fallback: direct text match for non-ontology roles
    if result['jd_category'] is None and jd_lower:
        jd_words = [w for w in jd_lower.split() if len(w) > 2]
        if jd_words:
            matched_words = sum(1 for w in jd_words if w in resume_lower)
            ratio = matched_words / len(jd_words)
            result['score'] = max(result['score'], int(ratio * 80))

    return result


# ==================== FORMATTING ====================

def calculate_formatting_penalty(text: str, sections: Dict[str, str]) -> Dict[str, Any]:
    """Calculate formatting penalty based on resume structure."""
    penalties = {
        'total': 0,
        'details': []
    }

    if 'skills' not in sections:
        penalties['total'] += SECTION_PENALTIES['missing_skills']
        penalties['details'].append('Missing skills section')

    if 'experience' not in sections:
        penalties['total'] += SECTION_PENALTIES['missing_experience']
        penalties['details'].append('Missing experience section')

    has_email = bool(re.search(r'[\w.-]+@[\w.-]+\.\w+', text))
    has_phone = bool(re.search(r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\./0-9]{7,}', text))

    if not (has_email or has_phone):
        penalties['total'] += SECTION_PENALTIES['missing_contact']
        penalties['details'].append('Missing contact information')

    if re.search(r'\b201x\b|\b20xx\b|\bxxxx\b', text.lower()):
        penalties['total'] += SECTION_PENALTIES['vague_dates']
        penalties['details'].append('Vague dates detected')

    bullet_count = len(re.findall(r'[•\-\*]\s', text))
    if bullet_count < 3:
        penalties['total'] += SECTION_PENALTIES['no_bullets']
        penalties['details'].append('Few bullet points (consider using more)')

    penalties['total'] = min(penalties['total'], 100)

    return penalties


# ==================== STUFFING DETECTION ====================

def detect_keyword_stuffing(resume_text: str, keywords: List[str]) -> Dict[str, Any]:
    """
    Detect keyword stuffing — abnormal repetition of JD keywords.

    Checks:
      1. Any single keyword repeated > MAX_KEYWORD_REPEATS times
      2. Total keyword density > MAX_KEYWORD_DENSITY of all tokens
    """
    result = {
        'is_stuffed': False,
        'penalty': 0,
        'flagged_keywords': [],
        'keyword_density': 0.0
    }

    if not keywords or not resume_text:
        return result

    text_lower = resume_text.lower()
    total_tokens = len(text_lower.split())
    if total_tokens == 0:
        return result

    total_keyword_hits = 0

    for kw in keywords:
        kw_lower = kw.lower()
        count = text_lower.count(kw_lower)
        total_keyword_hits += count

        if count > MAX_KEYWORD_REPEATS:
            result['flagged_keywords'].append(f"{kw} (×{count})")

    result['keyword_density'] = round(total_keyword_hits / total_tokens, 4)

    if result['flagged_keywords'] or result['keyword_density'] > MAX_KEYWORD_DENSITY:
        result['is_stuffed'] = True
        result['penalty'] = STUFFING_PENALTY

    return result


# ==================== RECENCY ====================

def calculate_recency_bonus(text: str) -> Dict[str, Any]:
    """
    Score recency of experience.

    Bonus for activity within the last 2 years, penalty for large gaps.
    """
    current_year = datetime.now().year
    years = extract_years_from_text(text)

    result = {
        'bonus': 0,
        'most_recent_year': None,
        'gap_years': 0,
        'details': ''
    }

    if not years:
        result['details'] = 'No dates found'
        return result

    most_recent = max(years)
    result['most_recent_year'] = most_recent
    gap = current_year - most_recent
    result['gap_years'] = gap

    if gap <= 1:
        result['bonus'] = 5
        result['details'] = 'Currently active or very recent experience'
    elif gap <= 2:
        result['bonus'] = 3
        result['details'] = 'Recent experience (within 2 years)'
    elif gap <= 4:
        result['bonus'] = 0
        result['details'] = 'Moderate gap in recent activity'
    else:
        result['bonus'] = -5
        result['details'] = f'Experience appears stale ({gap}+ year gap)'

    return result


# ==================== FIELD WEIGHTING ====================

def get_field_weight_for_keyword(keyword: str, sections: Dict[str, str]) -> float:
    """
    Determine how much weight a keyword gets based on WHERE it appears.

    Keywords in Experience bullets count 1.5×, in Skills 1.0×, etc.
    If found in multiple sections, the highest weight wins.
    """
    kw_lower = keyword.lower()
    best_weight = 0.0

    for section_name, section_text in sections.items():
        if kw_lower in section_text.lower():
            weight = SECTION_KEYWORD_WEIGHT.get(section_name, SECTION_KEYWORD_WEIGHT['_default'])
            best_weight = max(best_weight, weight)

    # If found in resume but not in any detected section
    if best_weight == 0.0:
        best_weight = SECTION_KEYWORD_WEIGHT['_default']

    return best_weight


# ==================== SCORE INTERPRETATION ====================

def get_score_interpretation(score: int) -> Dict[str, str]:
    """Get interpretation for ATS score."""
    if score >= 85:
        return {
            'badge': 'Excellent',
            'color': 'green',
            'message': 'Strong candidate - High match with job requirements'
        }
    elif score >= 70:
        return {
            'badge': 'Good',
            'color': 'blue',
            'message': 'Worth applying - Minor gaps to address'
        }
    elif score >= 50:
        return {
            'badge': 'Fair',
            'color': 'yellow',
            'message': 'Needs improvement - Notable gaps in requirements'
        }
    else:
        return {
            'badge': 'Poor',
            'color': 'red',
            'message': 'Major gaps - Consider other roles or significant resume updates'
        }
