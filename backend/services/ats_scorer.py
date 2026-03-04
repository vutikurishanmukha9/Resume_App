
import re
import json
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
from datetime import datetime
import os

# ==================== CONSTANTS ====================

# Scoring weights (configurable)
SCORING_WEIGHTS = {
    'skill_match': 0.40,
    'title_match': 0.20,
    'experience': 0.15,
    'achievement': 0.10,
    'education': 0.10,
    'formatting_penalty': 0.05
}

# Section penalties
SECTION_PENALTIES = {
    'missing_skills': 10,
    'missing_experience': 15,
    'missing_contact': 5,
    'vague_dates': 5,
    'no_bullets': 3
}

# Job title ontology for fuzzy matching
TITLE_ONTOLOGY = {
    # ── Tech / Software ──
    'software_engineer': [
        'software engineer', 'swe', 'software developer', 'developer',
        'programmer', 'software dev', 'application developer'
    ],
    'backend_engineer': [
        'backend engineer', 'backend developer', 'server-side developer',
        'back-end engineer', 'back end developer', 'api developer'
    ],
    'frontend_engineer': [
        'frontend engineer', 'frontend developer', 'front-end engineer',
        'front end developer', 'ui developer', 'ui engineer'
    ],
    'fullstack_engineer': [
        'fullstack engineer', 'full stack developer', 'full-stack engineer',
        'fullstack developer', 'full stack engineer'
    ],
    'data_scientist': [
        'data scientist', 'ml engineer', 'machine learning engineer',
        'ai engineer', 'data science engineer', 'applied scientist',
        'ai solutions engineer'
    ],
    'data_engineer': [
        'data engineer', 'etl developer', 'data pipeline engineer',
        'big data engineer', 'analytics engineer'
    ],
    'devops_engineer': [
        'devops engineer', 'sre', 'site reliability engineer',
        'platform engineer', 'infrastructure engineer', 'cloud engineer'
    ],
    'product_manager': [
        'product manager', 'pm', 'product owner', 'technical product manager',
        'associate product manager', 'apm'
    ],
    'data_analyst': [
        'data analyst', 'business analyst', 'analytics specialist',
        'bi analyst', 'business intelligence analyst'
    ],
    'qa_engineer': [
        'qa engineer', 'test engineer', 'quality assurance engineer',
        'sdet', 'automation engineer', 'test automation engineer'
    ],
    # ── Design / Creative ──
    'designer': [
        'ui/ux designer', 'ux designer', 'ui designer', 'product designer',
        'graphic designer', 'visual designer', 'interaction designer',
        'web designer'
    ],
    # ── Finance / Business ──
    'financial_analyst': [
        'financial analyst', 'finance analyst', 'investment analyst',
        'risk analyst', 'credit analyst', 'equity analyst',
        'portfolio analyst', 'treasury analyst'
    ],
    'accountant': [
        'accountant', 'cpa', 'auditor', 'tax analyst',
        'accounts payable', 'accounts receivable', 'bookkeeper'
    ],
    'consultant': [
        'consultant', 'management consultant', 'strategy consultant',
        'associate consultant', 'advisory', 'solutions consultant'
    ],
    # ── Healthcare ──
    'nurse': [
        'registered nurse', 'rn', 'nurse practitioner', 'clinical nurse',
        'staff nurse', 'charge nurse', 'lpn', 'nursing'
    ],
    'healthcare': [
        'physician', 'doctor', 'pharmacist', 'therapist',
        'medical assistant', 'healthcare administrator', 'clinical researcher'
    ],
    # ── Engineering (non-software) ──
    'mechanical_engineer': [
        'mechanical engineer', 'design engineer', 'manufacturing engineer',
        'process engineer', 'industrial engineer', 'quality engineer'
    ],
    'electrical_engineer': [
        'electrical engineer', 'electronics engineer', 'embedded engineer',
        'hardware engineer', 'control systems engineer', 'power engineer'
    ],
    'civil_engineer': [
        'civil engineer', 'structural engineer', 'construction engineer',
        'environmental engineer', 'geotechnical engineer', 'site engineer'
    ],
    # ── Marketing / Sales ──
    'marketing': [
        'marketing manager', 'digital marketing', 'seo specialist',
        'content strategist', 'brand manager', 'growth marketer',
        'social media manager', 'marketing analyst'
    ],
    'sales': [
        'sales representative', 'account executive', 'sales manager',
        'business development', 'sales engineer', 'account manager'
    ],
    # ── HR / Operations ──
    'hr': [
        'hr manager', 'human resources', 'recruiter', 'talent acquisition',
        'hr generalist', 'hr business partner', 'people operations'
    ],
    'project_manager': [
        'project manager', 'program manager', 'scrum master',
        'agile coach', 'delivery manager', 'technical project manager'
    ],
}

# Seniority levels
SENIORITY_PATTERNS = {
    'lead': ['lead', 'principal', 'staff', 'architect', 'head of', 'director'],
    'senior': ['senior', 'sr', 'sr.', 'experienced'],
    'mid': ['mid', 'mid-level', 'intermediate', 'ii', 'level 2'],
    'entry': ['junior', 'jr', 'jr.', 'entry', 'associate', 'trainee', 'intern', 'fresher', 'graduate']
}

# Education levels – use word-boundary patterns to avoid false positives
# (e.g. "Scrum Master" or "Mastered Python" should NOT match 'master')
EDUCATION_LEVELS = {
    'phd': ['phd', 'ph.d', 'doctorate', 'doctor of philosophy', 'doctoral'],
    'masters': ["master's", "master of", 'mba', 'm.s.', 'm.s ', 'm.tech', 'mtech',
                'msc', 'm.sc', 'ms in ', 'ma in ', 'postgraduate'],
    'bachelors': ["bachelor's", "bachelor of", 'b.tech', 'btech', 'b.s.', 'b.s ',
                  'b.e.', 'b.e ', 'bsc', 'b.sc', 'ba in ', 'bs in ',
                  'undergraduate degree'],
    'diploma': ['diploma in', 'associate degree', 'certification in', 'certificate in']
}

# Phrases that look like education keywords but aren't
EDUCATION_FALSE_POSITIVES = [
    'scrum master', 'master class', 'mastered', 'masters at',
    'master of none', 'webmaster', 'postmaster', 'taskmaster',
    'bachelor party',
]

# Keywords indicating requirement importance
REQUIRED_INDICATORS = [
    'must have', 'required', 'essential', 'mandatory', 'need to have',
    'you must', 'should have', 'necessary', 'prerequisite'
]

PREFERRED_INDICATORS = [
    'preferred', 'nice to have', 'bonus', 'plus', 'advantageous',
    'would be great', 'ideally', 'desirable', 'good to have'
]

# Negation phrases that invert requirement meaning
NEGATION_PHRASES = [
    'not required', 'not essential', 'not mandatory', 'not necessary',
    'no need', 'don\'t need', 'do not need',
    'not a must', 'not a requirement', 'optional',
]

# Action verbs for achievement detection
ACTION_VERBS = [
    'achieved', 'improved', 'reduced', 'increased', 'built', 'developed',
    'created', 'designed', 'implemented', 'launched', 'led', 'managed',
    'delivered', 'optimized', 'automated', 'scaled', 'grew', 'saved',
    'generated', 'streamlined', 'transformed', 'accelerated', 'enhanced'
]

# Section headers for resume parsing
SECTION_HEADERS = {
    'experience': ['experience', 'work experience', 'employment', 'professional experience', 
                   'work history', 'career history', 'professional background'],
    'education': ['education', 'academic', 'qualifications', 'academic background'],
    'skills': ['skills', 'technical skills', 'core competencies', 'technologies',
               'tools', 'expertise', 'proficiencies', 'competencies'],
    'projects': ['projects', 'personal projects', 'academic projects', 'portfolio'],
    'summary': ['summary', 'objective', 'profile', 'about', 'professional summary',
                'career objective', 'executive summary'],
    'certifications': ['certifications', 'certificates', 'credentials', 'licenses'],
    'achievements': ['achievements', 'accomplishments', 'awards', 'honors']
}


# ==================== HELPER FUNCTIONS ====================

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
    # (the feature extractor doesn't extract these)
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
                    # e.g. 'master' matches inside 'scrum master'
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
            # Look for header at start of line or after newline
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


def extract_achievements(text: str) -> List[Dict[str, Any]]:
    """
    Extract quantified achievements from text.
    
    Tightened rules:
    - Action verb must START the sentence/clause (not just appear anywhere)
    - Metric must be within 80 chars of the verb (same clause)
    - Filters out generic number mentions like "team of 5"
    """
    achievements = []
    
    # Split into sentences/bullets
    sentences = re.split(r'[.•\n]', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15:
            continue
        
        sentence_lower = sentence.lower().lstrip('- ')
        
        # Action verb must START the sentence/clause
        # (Verbs at the start indicate an achievement bullet, not a description)
        leading_verb = None
        for verb in ACTION_VERBS:
            if sentence_lower.startswith(verb):
                leading_verb = verb
                break
        
        if not leading_verb:
            continue
        
        # Check for metrics NEAR the verb (within the same clause)
        # This prevents "Optimized code" + "team of 5 people" false matches
        clause = sentence_lower[:120]  # first 120 chars only
        
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
    
    # Split JD into sentences for context
    sentences = re.split(r'[.•\n]', jd_lower)
    
    for keyword in all_keywords:
        keyword_lower = keyword.lower()
        classified = False
        
        for sentence in sentences:
            if keyword_lower in sentence:
                # Check for negation first — "not required" should NOT
                # flag the keyword as required
                has_negation = any(neg in sentence for neg in NEGATION_PHRASES)
                
                if has_negation:
                    # Negated requirement → treat as preferred at best
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
    
    # Fallback: if JD title doesn't match any ontology category,
    # do a direct text match so non-ontology roles aren't penalised to 0
    if result['jd_category'] is None and jd_lower:
        # Check if the raw JD title text appears anywhere in the resume
        jd_words = [w for w in jd_lower.split() if len(w) > 2]
        if jd_words:
            matched_words = sum(1 for w in jd_words if w in resume_lower)
            ratio = matched_words / len(jd_words)
            result['score'] = max(result['score'], int(ratio * 80))
    
    return result


def calculate_formatting_penalty(text: str, sections: Dict[str, str]) -> Dict[str, Any]:
    """
    Calculate formatting penalty based on resume structure.
    """
    penalties = {
        'total': 0,
        'details': []
    }
    
    # Check for missing sections
    if 'skills' not in sections:
        penalties['total'] += SECTION_PENALTIES['missing_skills']
        penalties['details'].append('Missing skills section')
    
    if 'experience' not in sections:
        penalties['total'] += SECTION_PENALTIES['missing_experience']
        penalties['details'].append('Missing experience section')
    
    # Check for contact info
    has_email = bool(re.search(r'[\w.-]+@[\w.-]+\.\w+', text))
    has_phone = bool(re.search(r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\./0-9]{7,}', text))
    
    if not (has_email or has_phone):
        penalties['total'] += SECTION_PENALTIES['missing_contact']
        penalties['details'].append('Missing contact information')
    
    # Check for vague dates
    if re.search(r'\b201x\b|\b20xx\b|\bxxxx\b', text.lower()):
        penalties['total'] += SECTION_PENALTIES['vague_dates']
        penalties['details'].append('Vague dates detected')
    
    # Check for bullet points (good structure)
    bullet_count = len(re.findall(r'[•\-\*]\s', text))
    if bullet_count < 3:
        penalties['total'] += SECTION_PENALTIES['no_bullets']
        penalties['details'].append('Few bullet points (consider using more)')
    
    # Cap at 100
    penalties['total'] = min(penalties['total'], 100)
    
    return penalties


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


def generate_suggestions(
    missing_keywords: Dict[str, List[str]],
    achievements: List[Dict],
    experience: Dict,
    sections: Dict[str, str],
    formatting_penalty: Dict,
    skill_score: int = 0,
    title_score: int = 0,
    education_score: int = 0,
    matched_skills: Dict[str, List[str]] = None
) -> List[str]:
    """
    Generate dynamic, context-aware improvement suggestions.
    
    Suggestions are prioritized and specific to the actual gaps found.
    """
    suggestions = []
    
    # Priority 1: Critical missing keywords (most impactful)
    critical_missing = missing_keywords.get('critical', [])
    if critical_missing:
        if len(critical_missing) >= 5:
            top_3 = critical_missing[:3]
            suggestions.append(
                f"CRITICAL: Add these required skills to your resume: {', '.join(top_3)} "
                f"(+{len(critical_missing) - 3} more missing)"
            )
        elif len(critical_missing) >= 2:
            suggestions.append(
                f"Add these required keywords: {', '.join(critical_missing)}"
            )
        else:
            suggestions.append(
                f"Add the required keyword '{critical_missing[0]}' to match the job description"
            )
    
    # Priority 2: Important missing keywords
    important_missing = missing_keywords.get('important', [])
    if important_missing and len(suggestions) < 4:
        top_important = important_missing[:3]
        suggestions.append(
            f"Consider adding preferred skills: {', '.join(top_important)}"
        )
    
    # Priority 3: Skill match suggestions (based on actual score)
    if skill_score is not None:
        if skill_score < 30:
            suggestions.append(
                "Your skill match is low. Review the job description and add relevant technologies you know"
            )
        elif skill_score < 50:
            # Suggest adding more matched skills context
            if matched_skills:
                matched_count = sum(len(v) for v in matched_skills.values())
                if matched_count < 5:
                    suggestions.append(
                        f"Only {matched_count} skills matched. List more of your technical skills explicitly"
                    )
    
    # Priority 4: Achievement suggestions (dynamic based on count)
    achievement_count = len(achievements)
    if achievement_count == 0:
        suggestions.append(
            "Add quantified achievements with metrics (%, $, numbers) to show impact"
        )
    elif achievement_count < 3:
        needed = 3 - achievement_count
        suggestions.append(
            f"Add {needed} more quantified achievements. Examples: 'Increased X by 25%', 'Reduced Y by $10K'"
        )
    
    # Priority 5: Experience suggestions (based on actual data)
    total_years = experience.get('total_years', 0)
    skill_years = experience.get('skill_years', {})
    
    if total_years == 0:
        suggestions.append(
            "Specify your total years of experience clearly (e.g., '5+ years of experience')"
        )
    elif not skill_years:
        suggestions.append(
            "Add years of experience for key skills (e.g., 'Python - 4 years', 'AWS - 2 years')"
        )
    
    # Priority 6: Section suggestions (specific to what's missing)
    missing_sections = []
    if 'skills' not in sections:
        missing_sections.append('Skills')
    if 'summary' not in sections:
        missing_sections.append('Professional Summary')
    if 'projects' not in sections and 'experience' not in sections:
        missing_sections.append('Projects or Experience')
    
    if missing_sections and len(suggestions) < 5:
        suggestions.append(
            f"Add missing sections: {', '.join(missing_sections)}"
        )
    
    # Priority 7: Title match suggestions
    if title_score is not None and title_score < 50:
        suggestions.append(
            "Include the target job title or related titles in your resume summary"
        )
    
    # Priority 8: Education suggestions
    if education_score is not None and education_score < 40:
        suggestions.append(
            "Clearly list your educational qualifications with degree names"
        )
    
    # Priority 9: Formatting suggestions (specific)
    for detail in formatting_penalty.get('details', []):
        if 'bullet' in detail.lower() and len(suggestions) < 6:
            suggestions.append(
                "Use more bullet points (aim for 3-5 per job) to highlight key accomplishments"
            )
        elif 'contact' in detail.lower() and len(suggestions) < 6:
            suggestions.append(
                "Add complete contact information: email, phone, LinkedIn profile"
            )
        elif 'date' in detail.lower() and len(suggestions) < 6:
            suggestions.append(
                "Use specific date ranges (e.g., 'Jan 2020 - Dec 2023') instead of vague dates"
            )
    
    # Priority 10: General improvement if score is still low
    if len(suggestions) < 3:
        # Add general tips if we don't have enough specific suggestions
        optional_missing = missing_keywords.get('optional', [])
        if optional_missing and len(optional_missing) > 3:
            suggestions.append(
                f"Bonus: Include optional skills like {', '.join(optional_missing[:2])} if applicable"
            )
    
    # Return top 6 most impactful suggestions
    return suggestions[:6]



# ==================== MAIN SCORER CLASS ====================

class ATSScorer:
    """
    ATS Score Calculator with configurable weights and dual analysis modes.
    """
    
    def __init__(self, mode: str = "deep", weights: Dict[str, float] = None):
        """
        Initialize ATS Scorer.
        
        Args:
            mode: "quick" for fast keyword matching, "deep" for full analysis
            weights: Custom scoring weights (optional)
        """
        self.mode = mode
        self.weights = weights or SCORING_WEIGHTS.copy()
    
    def calculate_skill_score(
        self,
        resume_skills: List[str],
        jd_keywords: Dict[str, List[str]],
        semantic_similarity: float = None
    ) -> Dict[str, Any]:
        """Calculate skill match score."""
        resume_skills_lower = [s.lower() for s in resume_skills]
        
        matched = {
            'required': [],
            'preferred': [],
            'standard': []
        }
        missing = {
            'critical': [],
            'important': [],
            'optional': []
        }
        
        # Check required keywords (weight: 2x)
        for keyword in jd_keywords.get('required', []):
            if keyword.lower() in resume_skills_lower:
                matched['required'].append(keyword)
            else:
                missing['critical'].append(keyword)
        
        # Check preferred keywords (weight: 1x)
        for keyword in jd_keywords.get('preferred', []):
            if keyword.lower() in resume_skills_lower:
                matched['preferred'].append(keyword)
            else:
                missing['important'].append(keyword)
        
        # Check standard keywords
        for keyword in jd_keywords.get('standard', []):
            if keyword.lower() in resume_skills_lower:
                matched['standard'].append(keyword)
            else:
                missing['optional'].append(keyword)
        
        # Calculate score
        total_required = len(jd_keywords.get('required', []))
        total_preferred = len(jd_keywords.get('preferred', []))
        total_standard = len(jd_keywords.get('standard', []))
        
        # Weighted score calculation
        required_score = (len(matched['required']) / max(total_required, 1)) * 50
        preferred_score = (len(matched['preferred']) / max(total_preferred, 1)) * 30
        standard_score = (len(matched['standard']) / max(total_standard, 1)) * 20
        
        base_score = required_score + preferred_score + standard_score
        
        # Add semantic similarity boost in deep mode
        if self.mode == "deep" and semantic_similarity is not None:
            semantic_boost = semantic_similarity * 10  # Up to 10 point boost
            base_score = min(100, base_score + semantic_boost)
        
        return {
            'score': int(base_score),
            'matched': matched,
            'missing': missing
        }
    
    def calculate_title_score(self, resume_text: str, jd_title: str) -> Dict[str, Any]:
        """Calculate job title match score."""
        title_match = match_job_title(resume_text, jd_title)
        seniority = detect_seniority_level(resume_text)
        
        # Combine title category match and seniority
        base_score = title_match['score']
        
        # Add seniority bonus (up to 20 points)
        seniority_bonus = seniority['level_score'] * 5
        
        return {
            'score': min(100, base_score + seniority_bonus),
            'title_match': title_match,
            'seniority': seniority
        }
    
    def calculate_experience_score(
        self,
        resume_experience: Dict,
        required_years: int = 0
    ) -> Dict[str, Any]:
        """Calculate experience score."""
        total_years = resume_experience.get('total_years', 0)
        
        if required_years > 0:
            # Score based on how well candidate meets requirement
            ratio = total_years / required_years
            score = min(100, int(ratio * 100))
        else:
            # Default scoring based on experience
            if total_years >= 10:
                score = 100
            elif total_years >= 5:
                score = 80
            elif total_years >= 3:
                score = 60
            elif total_years >= 1:
                score = 40
            else:
                score = 20
        
        return {
            'score': score,
            'total_years': total_years,
            'skill_years': resume_experience.get('skill_years', {})
        }
    
    def calculate_achievement_score(self, achievements: List[Dict]) -> Dict[str, Any]:
        """Calculate achievement score based on quantified accomplishments."""
        count = len(achievements)
        
        if count >= 5:
            score = 100
        elif count >= 3:
            score = 75
        elif count >= 1:
            score = 50
        else:
            score = 20
        
        return {
            'score': score,
            'count': count,
            'achievements': achievements[:5]  # Return top 5
        }
    
    def calculate_education_score(
        self,
        resume_education: Dict,
        required_level: str = None
    ) -> Dict[str, Any]:
        """Calculate education score."""
        level = resume_education.get('highest_level', 'unknown')
        level_score = resume_education.get('level_score', 0)
        
        if required_level:
            required_score = {'phd': 4, 'masters': 3, 'bachelors': 2, 'diploma': 1}.get(required_level.lower(), 0)
            if level_score >= required_score:
                score = 100
            elif level_score == required_score - 1:
                score = 70
            else:
                score = 40
        else:
            # Default scoring
            score = level_score * 25  # 0-100
        
        return {
            'score': max(score, 20),  # Minimum 20
            'level': level,
            'degrees_found': resume_education.get('degrees_found', [])
        }
    
    def calculate_ats_score(
        self,
        resume_text: str,
        jd_text: str,
        jd_title: str = "",
        required_years: int = 0,
        required_education: str = None,
        resume_skills: List[str] = None,
        jd_keywords: List[str] = None,
        semantic_similarity: float = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive ATS score.
        
        Args:
            resume_text: Full resume text
            jd_text: Job description text
            jd_title: Job title (optional)
            required_years: Required years of experience (optional)
            required_education: Required education level (optional)
            resume_skills: Pre-extracted resume skills (optional)
            jd_keywords: Pre-extracted JD keywords (optional)
            semantic_similarity: Pre-computed semantic similarity (optional)
        
        Returns:
            Comprehensive score breakdown with suggestions
        """
        # Quick mode: keyword matching only
        if self.mode == "quick":
            return self._quick_scan(resume_text, jd_text, resume_skills, jd_keywords)
        
        # Deep mode: full analysis
        return self._deep_analysis(
            resume_text, jd_text, jd_title,
            required_years, required_education,
            resume_skills, jd_keywords, semantic_similarity
        )
    
    def _quick_scan(
        self,
        resume_text: str,
        jd_text: str,
        resume_skills: List[str] = None,
        jd_keywords: List[str] = None
    ) -> Dict[str, Any]:
        """Quick keyword-matching scan."""
        # Basic keyword extraction if not provided
        if not resume_skills:
            resume_skills = re.findall(r'\b\w+\b', resume_text.lower())
        if not jd_keywords:
            jd_keywords = re.findall(r'\b\w+\b', jd_text.lower())
        
        # Simple overlap calculation
        resume_set = set(resume_skills)
        jd_set = set(jd_keywords)
        
        matched = list(resume_set & jd_set)
        missing = list(jd_set - resume_set)
        
        match_ratio = len(matched) / max(len(jd_set), 1)
        score = int(match_ratio * 100)
        
        return {
            'ats_score': score,
            'mode': 'quick',
            'interpretation': get_score_interpretation(score),
            'matched_keywords': matched[:20],
            'missing_keywords': {'critical': missing[:10], 'important': [], 'optional': []},
            'suggestions': ['Run Deep Analysis for detailed breakdown']
        }
    
    def _deep_analysis(
        self,
        resume_text: str,
        jd_text: str,
        jd_title: str,
        required_years: int,
        required_education: str,
        resume_skills: List[str],
        jd_keywords: List[str],
        semantic_similarity: float
    ) -> Dict[str, Any]:
        """Full deep analysis with all sub-scores."""
        
        # Detect resume sections
        sections = detect_resume_sections(resume_text)
        
        # Extract experience
        experience = extract_experience_duration(resume_text)
        
        # Detect education
        education = detect_education_level(resume_text)
        
        # Extract achievements
        achievements = extract_achievements(resume_text)
        
        # Classify JD keywords
        if jd_keywords:
            classified_keywords = classify_jd_keywords(jd_text, jd_keywords)
        else:
            classified_keywords = {'required': [], 'preferred': [], 'standard': []}
        
        # Calculate formatting penalty
        formatting = calculate_formatting_penalty(resume_text, sections)
        
        # Calculate sub-scores
        skill_result = self.calculate_skill_score(
            resume_skills or [], classified_keywords, semantic_similarity
        )
        
        title_result = self.calculate_title_score(resume_text, jd_title)
        
        experience_result = self.calculate_experience_score(experience, required_years)
        
        achievement_result = self.calculate_achievement_score(achievements)
        
        education_result = self.calculate_education_score(education, required_education)
        
        # Calculate final weighted score
        final_score = (
            self.weights['skill_match'] * skill_result['score'] +
            self.weights['title_match'] * title_result['score'] +
            self.weights['experience'] * experience_result['score'] +
            self.weights['achievement'] * achievement_result['score'] +
            self.weights['education'] * education_result['score'] -
            self.weights['formatting_penalty'] * formatting['total']
        )
        
        final_score = max(0, min(100, int(final_score)))
        
        # Generate dynamic suggestions with context
        suggestions = generate_suggestions(
            skill_result['missing'],
            achievements,
            experience,
            sections,
            formatting,
            skill_score=skill_result['score'],
            title_score=title_result['score'],
            education_score=education_result['score'],
            matched_skills=skill_result['matched']
        )
        
        # Build matched keywords with importance
        matched_keywords = []
        for kw in skill_result['matched'].get('required', []):
            matched_keywords.append({'keyword': kw, 'importance': 'required'})
        for kw in skill_result['matched'].get('preferred', []):
            matched_keywords.append({'keyword': kw, 'importance': 'preferred'})
        for kw in skill_result['matched'].get('standard', [])[:10]:
            matched_keywords.append({'keyword': kw, 'importance': 'standard'})
        
        return {
            'ats_score': final_score,
            'mode': 'deep',
            'interpretation': get_score_interpretation(final_score),
            'sub_scores': {
                'skill_match': skill_result['score'],
                'title_match': title_result['score'],
                'experience': experience_result['score'],
                'achievement': achievement_result['score'],
                'education': education_result['score'],
                'formatting_penalty': formatting['total']
            },
            'matched_keywords': matched_keywords,
            'missing_keywords': skill_result['missing'],
            'achievements_found': [a['text'] for a in achievement_result['achievements']],
            'experience_details': {
                'total_years': experience_result['total_years'],
                'skill_years': experience_result['skill_years']
            },
            'education_details': {
                'level': education_result['level'],
                'degrees': education_result['degrees_found']
            },
            'sections_detected': list(sections.keys()),
            'suggestions': suggestions
        }


# Convenience function for external use
def calculate_ats_score(
    resume_text: str,
    jd_text: str,
    mode: str = "deep",
    **kwargs
) -> Dict[str, Any]:
    """
    Calculate ATS score for resume against job description.
    
    Args:
        resume_text: Full resume text
        jd_text: Job description text
        mode: "quick" or "deep" analysis
        **kwargs: Additional parameters passed to ATSScorer
    
    Returns:
        Comprehensive ATS score breakdown
    """
    scorer = ATSScorer(mode=mode)
    return scorer.calculate_ats_score(resume_text, jd_text, **kwargs)
