"""
Keyword Extraction Utilities for AI Resume Analyzer

Handles keyword extraction with N-gram support, TF-IDF weighting,
keyword overlap calculation, and sentence splitting.
"""

import re
from typing import List, Dict
from collections import Counter


def extract_keywords(text: str, min_length: int = 3, use_ngrams: bool = True) -> List[str]:
    """
    Extract meaningful keywords from text with N-gram support and stemming.
    
    Args:
        text: Input text to extract keywords from
        min_length: Minimum word length to consider
        use_ngrams: Whether to include bigrams and trigrams
    
    Returns:
        List of keywords including single words and N-grams
    """
    # Convert to lowercase and clean
    text_clean = re.sub(r'[^a-zA-Z0-9\s+#\-/]', ' ', text.lower())
    
    # Common stop words to exclude
    stop_words = {
        'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was', 'were',
        'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'who', 'when',
        'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'just', 'but', 'for', 'with', 'about', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
        'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'or', 'and',
        'also', 'our', 'your', 'their', 'any', 'new', 'using', 'used', 'use', 'work',
        'working', 'worked', 'including', 'include', 'includes', 'ensure', 'ability'
    }
    
    # Extract words
    words = text_clean.split()
    
    # Apply simple stemming (suffix stripping)
    def simple_stem(word: str) -> str:
        """Simple suffix-based stemming"""
        suffixes = ['ing', 'ed', 'ly', 'ment', 'tion', 'sion', 'ness', 'able', 'ible']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[:-len(suffix)]
        return word
    
    # Filter and stem single words
    single_keywords = []
    for word in words:
        if len(word) >= min_length and word not in stop_words:
            stemmed = simple_stem(word)
            if len(stemmed) >= min_length:
                single_keywords.append(word)  # Keep original for matching
    
    keywords = single_keywords.copy()
    
    # Generate N-grams (bigrams and trigrams)
    if use_ngrams and len(words) >= 2:
        # Important technical bigrams to look for
        important_bigrams = {
            'machine learning', 'deep learning', 'data science', 'data analysis',
            'project management', 'software development', 'web development',
            'full stack', 'front end', 'back end', 'cloud computing',
            'artificial intelligence', 'natural language', 'computer vision',
            'big data', 'data engineering', 'data pipeline', 'api development',
            'mobile development', 'system design', 'database management',
            'version control', 'continuous integration', 'continuous deployment',
            'test driven', 'behavior driven', 'object oriented', 'functional programming',
            'agile methodology', 'scrum master', 'product management', 'team lead',
            'senior engineer', 'junior developer', 'software architect', 'tech lead',
            'ci cd', 'rest api', 'graphql api', 'microservices architecture'
        }
        
        # Important trigrams
        important_trigrams = {
            'machine learning engineer', 'data science team', 'full stack developer',
            'senior software engineer', 'junior software developer', 'natural language processing',
            'continuous integration deployment', 'test driven development', 'object oriented programming',
            'amazon web services', 'google cloud platform', 'microsoft azure cloud'
        }
        
        # Generate bigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if bigram in important_bigrams or (
                words[i] not in stop_words and 
                words[i+1] not in stop_words and
                len(words[i]) >= 2 and len(words[i+1]) >= 2
            ):
                keywords.append(bigram)
        
        # Generate trigrams
        for i in range(len(words) - 2):
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            if trigram in important_trigrams:
                keywords.append(trigram)
    
    return keywords


def calculate_tfidf_weights(keywords: List[str], corpus_keywords: List[str] = None) -> Dict[str, float]:
    """
    Calculate TF-IDF-like weights for keywords.
    Higher weight = more important/rare term.
    
    Args:
        keywords: Keywords from the current document
        corpus_keywords: Keywords from comparison document (used for IDF approximation)
    
    Returns:
        Dictionary mapping keywords to their weights
    """
    # Term frequency
    tf = Counter(keywords)
    total_terms = len(keywords)
    
    # Common terms that should have lower weight
    common_terms = {
        'experience', 'years', 'work', 'team', 'skills', 'knowledge',
        'strong', 'excellent', 'good', 'great', 'best', 'required',
        'preferred', 'must', 'responsibilities', 'requirements', 'role',
        'position', 'job', 'company', 'organization', 'looking', 'seeking',
        'candidate', 'candidates', 'application', 'apply', 'email'
    }
    
    # Technical terms that should have higher weight
    technical_terms = {
        'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue',
        'node', 'django', 'flask', 'spring', 'docker', 'kubernetes', 'aws',
        'azure', 'gcp', 'sql', 'nosql', 'mongodb', 'postgresql', 'redis',
        'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit',
        'data science', 'data engineering', 'api', 'rest', 'graphql', 'microservices',
        'ci/cd', 'devops', 'agile', 'scrum', 'git', 'linux', 'algorithms'
    }
    
    weights = {}
    for keyword, count in tf.items():
        # Base TF weight (normalized)
        base_weight = count / total_terms if total_terms > 0 else 0
        
        # Apply multipliers based on term importance
        keyword_lower = keyword.lower()
        
        if keyword_lower in technical_terms or any(tech in keyword_lower for tech in technical_terms):
            # Technical terms get 3x weight
            weights[keyword] = base_weight * 3.0
        elif keyword_lower in common_terms:
            # Common terms get reduced weight
            weights[keyword] = base_weight * 0.3
        elif len(keyword.split()) >= 2:
            # N-grams get higher weight (more specific)
            weights[keyword] = base_weight * 2.0
        else:
            weights[keyword] = base_weight * 1.0
    
    # Normalize weights to 0-1 range
    if weights:
        max_weight = max(weights.values())
        if max_weight > 0:
            weights = {k: v / max_weight for k, v in weights.items()}
    
    return weights


def calculate_keyword_overlap(resume_keywords: List[str], jd_keywords: List[str]) -> float:
    """
    Calculate keyword overlap between resume and JD using TF-IDF weighting.
    Technical and N-gram keywords get higher weight.
    """
    if not jd_keywords:
        return 0.0
    
    # Get TF-IDF weights for JD keywords (these are what matter for matching)
    jd_weights = calculate_tfidf_weights(jd_keywords)
    
    # Normalize keywords for comparison
    resume_set = set(kw.lower() for kw in resume_keywords)
    
    # Calculate weighted overlap
    matched_weight = 0.0
    total_weight = 0.0
    
    for keyword in jd_keywords:
        keyword_lower = keyword.lower()
        weight = jd_weights.get(keyword, 0.1)  # Default low weight if not found
        total_weight += weight
        
        # Check for exact match
        if keyword_lower in resume_set:
            matched_weight += weight
        else:
            # Check for partial match (keyword is part of a longer phrase in resume)
            for resume_kw in resume_set:
                if keyword_lower in resume_kw or resume_kw in keyword_lower:
                    matched_weight += weight * 0.7  # Partial match gets 70% credit
                    break
    
    if total_weight == 0:
        return 0.0
    
    # Calculate percentage and apply calibration
    raw_score = (matched_weight / total_weight) * 100
    
    # Calibrate: boost low scores slightly, cap at 100
    # This helps when there's partial but meaningful overlap
    calibrated_score = min(100, raw_score * 1.2)
    
    return calibrated_score


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def get_missing_keywords(resume_keywords: List[str], jd_keywords: List[str]) -> Dict[str, List[str]]:
    """
    Identify keywords present in JD but missing from resume.
    Returns keywords categorized by importance (critical, important, optional).
    """
    if not jd_keywords:
        return {'critical': [], 'important': [], 'optional': []}
    
    resume_set = set(kw.lower() for kw in resume_keywords)
    jd_counter = Counter(jd_keywords)
    
    missing = []
    for keyword, count in jd_counter.items():
        if keyword.lower() not in resume_set:
            missing.append((keyword, count))
    
    # Sort by frequency
    missing.sort(key=lambda x: x[1], reverse=True)
    
    # Categorize by frequency
    total_missing = len(missing)
    if total_missing == 0:
        return {'critical': [], 'important': [], 'optional': []}
    
    # Top 30% are critical, next 40% are important, rest are optional
    critical_count = max(1, int(total_missing * 0.3))
    important_count = max(1, int(total_missing * 0.4))
    
    categorized = {
        'critical': [kw for kw, _ in missing[:critical_count]],
        'important': [kw for kw, _ in missing[critical_count:critical_count + important_count]],
        'optional': [kw for kw, _ in missing[critical_count + important_count:]]
    }
    
    return categorized
