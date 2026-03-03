"""
Feature Extraction Utilities for AI Resume Analyzer

Extracts experience years, education level, seniority level,
and combined resume features for salary prediction.
"""

import re
from typing import Dict, Any
from datetime import datetime

from backend.utils.skill_extractor import extract_skills, get_all_skills_flat


def extract_years_of_experience(text: str) -> float:
    """
    Extract years of experience from resume text.
    Looks for patterns like "X years", "X+ years", date ranges, etc.
    Uses dynamic current year for accurate calculation.
    """
    text_lower = text.lower()
    years = []
    
    # Get current year dynamically
    current_year = datetime.now().year
    
    # Pattern 1: "X years of experience" or "X+ years"
    pattern1 = r'(\d+)\+?\s*years?\s*(?:of)?\s*(?:experience|exp|expertise)'
    matches1 = re.findall(pattern1, text_lower)
    years.extend([int(m) for m in matches1])
    
    # Pattern 2: "X-Y years of experience"
    pattern2 = r'(\d+)\s*[-–]\s*(\d+)\s*years?\s*(?:of)?\s*(?:experience|exp)'
    matches2 = re.findall(pattern2, text_lower)
    for min_yr, max_yr in matches2:
        # Take the average or max of the range
        years.append(int(max_yr))
    
    # Pattern 3: Extract years from date ranges
    # Format: "2019 - 2023", "Jan 2020 - Present", "2018-present"
    year_pattern = r'\b((?:19|20)\d{2})\b'
    year_matches = re.findall(year_pattern, text)
    
    # Check for "Present" or "Current" to include current year
    if re.search(r'\b(present|current|till date|to date|ongoing)\b', text_lower):
        year_matches.append(str(current_year))
    
    if len(year_matches) >= 2:
        # Calculate experience from date ranges
        years_found = list(set([int(y) for y in year_matches]))  # Unique years
        years_found.sort()
        
        if years_found:
            max_year = max(years_found)
            min_year = min(years_found)
            
            # If max year is in the future, cap it at current year
            if max_year > current_year:
                max_year = current_year
            
            calculated_exp = max_year - min_year
            if 0 < calculated_exp < 50:  # Sanity check
                years.append(calculated_exp)
    
    # Pattern 4: Month-Year to Month-Year format
    # "Jan 2019 - Dec 2022", "January 2020 - Present"
    date_range_pattern = r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*(20\d{2})\s*[-–]\s*(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*)?(20\d{2}|present|current)'
    date_ranges = re.findall(date_range_pattern, text_lower)
    for start_year, end in date_ranges:
        end_year = current_year if end in ['present', 'current'] else int(end)
        exp = end_year - int(start_year)
        if 0 < exp < 50:
            years.append(exp)
    
    # Return maximum years found (most likely to be total experience)
    if years:
        return float(max(years))
    
    # Default to 0 if no experience found
    return 0.0


def extract_education_level(text: str) -> int:
    """
    Extract education level from resume.
    Returns: 0=Unknown, 1=Bachelor's, 2=Master's, 3=PhD
    """
    text_lower = text.lower()
    
    # Check for PhD/Doctorate
    phd_patterns = [r'\bph\.?d\b', r'\bdoctorate\b', r'\bdoctoral\b']
    for pattern in phd_patterns:
        if re.search(pattern, text_lower):
            return 3
    
    # Check for Master's
    masters_patterns = [r'\bmaster', r'\bm\.?s\.?\b', r'\bmba\b', r'\bm\.?tech\b', r'\bm\.?sc\b']
    for pattern in masters_patterns:
        if re.search(pattern, text_lower):
            return 2
    
    # Check for Bachelor's
    bachelors_patterns = [r'\bbachelor', r'\bb\.?s\.?\b', r'\bb\.?tech\b', r'\bb\.?sc\b', r'\bb\.?e\.?\b', r'\bundergraduate\b']
    for pattern in bachelors_patterns:
        if re.search(pattern, text_lower):
            return 1
    
    return 0  # Unknown


def extract_seniority_level(text: str) -> int:
    """
    Extract job seniority level from resume.
    Returns: 0=Entry, 1=Mid, 2=Senior, 3=Lead/Principal
    """
    text_lower = text.lower()
    
    # Check for Lead/Principal/Director level
    lead_patterns = [r'\blead\b', r'\bprincipal\b', r'\bdirector\b', r'\bhead of\b', r'\bvp\b', r'\bchief\b']
    for pattern in lead_patterns:
        if re.search(pattern, text_lower):
            return 3
    
    # Check for Senior level
    senior_patterns = [r'\bsenior\b', r'\bsr\.?\b', r'\bstaff\b']
    for pattern in senior_patterns:
        if re.search(pattern, text_lower):
            return 2
    
    # Check for Mid level
    mid_patterns = [r'\bmid-level\b', r'\bintermediate\b', r'\bassociate\b']
    for pattern in mid_patterns:
        if re.search(pattern, text_lower):
            return 1
    
    # Check for Junior/Entry level
    junior_patterns = [r'\bjunior\b', r'\bjr\.?\b', r'\bentry\b', r'\bintern\b', r'\btrainee\b']
    for pattern in junior_patterns:
        if re.search(pattern, text_lower):
            return 0
    
    # Default to mid-level if no clear indicator
    return 1


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
