"""
Skill Extraction Utilities for AI Resume Analyzer

Handles skills taxonomy matching, basic skill extraction fallback,
and skill matching calculations.
"""

import re
import json
import logging
from typing import Dict

from backend.config import SKILLS_TAXONOMY_PATH

logger = logging.getLogger(__name__)


def extract_skills(text: str) -> Dict[str, set]:
    """
    Extract technical skills and technologies from text using comprehensive taxonomy.
    Returns categorized skills with normalized names.
    """
    text_lower = text.lower()
    
    # Load skills taxonomy
    try:
        with open(SKILLS_TAXONOMY_PATH, 'r', encoding='utf-8') as f:
            taxonomy = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load skills taxonomy: {e}, using basic extraction")
        # Fallback to basic extraction
        return _extract_skills_basic(text_lower)
    
    # Initialize categorized skills
    categorized_skills = {
        'programming_languages': set(),
        'web_frameworks': set(),
        'databases': set(),
        'cloud_platforms': set(),
        'devops_tools': set(),
        'data_science_ml': set(),
        'mobile_development': set(),
        'testing_frameworks': set(),
        'other_technologies': set(),
        'methodologies': set(),
        'soft_skills': set(),
        'design_tools': set(),
        'other_tools': set()
    }
    
    # Extract skills from each category
    for category, skills_dict in taxonomy.items():
        if category not in categorized_skills:
            continue
            
        for canonical_skill, variations in skills_dict.items():
            # Check if any variation is present in the text
            for variation in variations:
                # Escape special regex characters in variation
                escaped_variation = re.escape(variation)
                # Use word boundaries for accurate matching
                pattern = r'\b' + escaped_variation + r'\b'
                
                if re.search(pattern, text_lower, re.IGNORECASE):
                    # Add canonical skill name
                    categorized_skills[category].add(canonical_skill)
                    break  # Found this skill, no need to check other variations
    
    return categorized_skills


def _extract_skills_basic(text_lower: str) -> Dict[str, set]:
    """Fallback basic skills extraction if taxonomy file is not available"""
    skill_patterns = {
        'programming_languages': r'\b(python|java|javascript|typescript|c\+\+|c#|ruby|php|swift|kotlin|go|rust|scala|r|matlab)\b',
        'web_frameworks': r'\b(html|css|react|angular|vue|node\.?js|express|django|flask|spring|asp\.net)\b',
        'databases': r'\b(sql|mysql|postgresql|mongodb|redis|oracle|sqlite|cassandra|dynamodb)\b',
        'cloud_platforms': r'\b(aws|azure|gcp)\b',
        'devops_tools': r'\b(docker|kubernetes|jenkins|terraform|ansible|git|github|gitlab)\b',
        'data_science_ml': r'\b(machine learning|deep learning|tensorflow|pytorch|scikit-learn|pandas|numpy|matplotlib|nlp|computer vision)\b',
        'other_technologies': r'\b(rest|api|microservices|ci/cd|linux|unix|windows|bash|powershell)\b',
        'methodologies': r'\b(agile|scrum)\b',
        'soft_skills': r'\b(leadership|communication|problem solving|team work|analytical|project management)\b'
    }
    
    categorized_skills = {}
    for category, pattern in skill_patterns.items():
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        categorized_skills[category] = set(matches)
    
    return categorized_skills


def get_all_skills_flat(categorized_skills: Dict[str, set]) -> set:
    """Flatten categorized skills into a single set for backward compatibility"""
    all_skills = set()
    for skills_set in categorized_skills.values():
        all_skills.update(skills_set)
    return all_skills


def calculate_skills_match(resume_skills: set, jd_skills: set) -> float:
    """Calculate skills matching percentage"""
    if not jd_skills:
        return 100.0  # If no specific skills in JD, give full score
    
    matching_skills = resume_skills & jd_skills
    return (len(matching_skills) / len(jd_skills)) * 100
