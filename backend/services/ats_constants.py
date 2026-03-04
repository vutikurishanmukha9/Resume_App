"""
ATS Scorer Constants

All dictionaries, patterns, weights, and thresholds used by the
ATS scoring engine. Separated for clarity and maintainability.
"""

# ==================== SCORING WEIGHTS ====================

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

# ==================== JOB TITLE ONTOLOGY ====================

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

# ==================== PATTERNS ====================

SENIORITY_PATTERNS = {
    'lead': ['lead', 'principal', 'staff', 'architect', 'head of', 'director'],
    'senior': ['senior', 'sr', 'sr.', 'experienced'],
    'mid': ['mid', 'mid-level', 'intermediate', 'ii', 'level 2'],
    'entry': ['junior', 'jr', 'jr.', 'entry', 'associate', 'trainee', 'intern', 'fresher', 'graduate']
}

# Education levels – use word-boundary patterns to avoid false positives
EDUCATION_LEVELS = {
    'phd': ['phd', 'ph.d', 'doctorate', 'doctor of philosophy', 'doctoral'],
    'masters': ["master's", "master of", 'mba', 'm.s.', 'm.s ', 'm.tech', 'mtech',
                'msc', 'm.sc', 'ms in ', 'ma in ', 'postgraduate'],
    'bachelors': ["bachelor's", "bachelor of", 'b.tech', 'btech', 'b.s.', 'b.s ',
                  'b.e.', 'b.e ', 'bsc', 'b.sc', 'ba in ', 'bs in ',
                  'undergraduate degree'],
    'diploma': ['diploma in', 'associate degree', 'certification in', 'certificate in']
}

EDUCATION_FALSE_POSITIVES = [
    'scrum master', 'master class', 'mastered', 'masters at',
    'master of none', 'webmaster', 'postmaster', 'taskmaster',
    'bachelor party',
]

# ==================== KEYWORD CLASSIFICATION ====================

REQUIRED_INDICATORS = [
    'must have', 'required', 'essential', 'mandatory', 'need to have',
    'you must', 'should have', 'necessary', 'prerequisite'
]

PREFERRED_INDICATORS = [
    'preferred', 'nice to have', 'bonus', 'plus', 'advantageous',
    'would be great', 'ideally', 'desirable', 'good to have'
]

NEGATION_PHRASES = [
    'not required', 'not essential', 'not mandatory', 'not necessary',
    'no need', "don't need", 'do not need',
    'not a must', 'not a requirement', 'optional',
]

# ==================== ACHIEVEMENT DETECTION ====================

ACTION_VERBS = [
    'achieved', 'improved', 'reduced', 'increased', 'built', 'developed',
    'created', 'designed', 'implemented', 'launched', 'led', 'managed',
    'delivered', 'optimized', 'automated', 'scaled', 'grew', 'saved',
    'generated', 'streamlined', 'transformed', 'accelerated', 'enhanced'
]

# ==================== SECTION HEADERS ====================

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

# ==================== FIELD WEIGHTS ====================

# Keywords in experience bullets count more than in a plain skills list.
# Real ATS systems apply this weighting — proof in context beats a keyword list.
SECTION_KEYWORD_WEIGHT = {
    'experience':     1.5,
    'projects':       1.3,
    'achievements':   1.3,
    'summary':        1.2,
    'certifications': 1.1,
    'skills':         1.0,
    'education':      0.8,
    '_default':       0.7,   # keyword found but not in any detected section
}

# ==================== STUFFING & FREQUENCY ====================

MAX_KEYWORD_REPEATS = 5       # any single keyword appearing more than this is suspicious
MAX_KEYWORD_DENSITY = 0.03    # total keyword tokens / total tokens > 3% is suspicious
STUFFING_PENALTY = 15         # points deducted for detected stuffing

# Keyword frequency bonus: mentioning a keyword 2-3× shows contextual usage.
# Beyond 4 is diminishing, and 5+ triggers stuffing detection.
FREQUENCY_BONUS_PER_EXTRA = 0.15   # bonus weight multiplier per extra mention (beyond 1)
FREQUENCY_BONUS_MAX_MENTIONS = 4   # cap: only count up to 4 mentions
FREQUENCY_BONUS_MAX_TOTAL = 10     # max total bonus points from frequency
