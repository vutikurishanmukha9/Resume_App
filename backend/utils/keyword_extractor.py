"""
Keyword Extraction Utilities for AI Resume Analyzer

Handles keyword extraction with N-gram support, TF-IDF weighting,
keyword overlap calculation, and sentence splitting.
"""

import re
from typing import List, Dict, Set
from collections import Counter


# ────────────────────────────────────────────────────────────────────
# STOP WORDS – comprehensive list of words that carry no meaning
# ────────────────────────────────────────────────────────────────────
STOP_WORDS: Set[str] = {
    # articles / determiners
    'the', 'a', 'an', 'this', 'that', 'these', 'those',
    # pronouns
    'i', 'me', 'my', 'you', 'your', 'he', 'she', 'it', 'its', 'we', 'us',
    'our', 'they', 'them', 'their', 'who', 'whom', 'what', 'which',
    # prepositions / conjunctions
    'at', 'on', 'in', 'to', 'for', 'with', 'about', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down',
    'out', 'off', 'over', 'under', 'between', 'within', 'of', 'by',
    'or', 'and', 'but', 'nor', 'so', 'yet', 'both', 'either', 'neither',
    'along', 'among', 'around', 'upon', 'across', 'toward', 'towards',
    # be / have / do / modals
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
    'has', 'have', 'had', 'having',
    'do', 'does', 'did', 'doing',
    'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could',
    # adverbs / adjectives / fillers
    'not', 'no', 'very', 'too', 'also', 'just', 'only', 'more', 'most',
    'less', 'then', 'than', 'once', 'again', 'further', 'ever', 'never',
    'now', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'each', 'every', 'few', 'some', 'such', 'own', 'same', 'other',
    'well', 'much', 'even', 'still', 'already', 'often', 'always',
    # generic resume / JD filler words
    'experience', 'experiences', 'years', 'year',
    'work', 'working', 'worked', 'works',
    'include', 'includes', 'including', 'included',
    'ensure', 'ensuring', 'ensured',
    'ability', 'able', 'capable',
    'strong', 'excellent', 'good', 'great', 'best', 'proven', 'adept',
    'skilled', 'proficient', 'demonstrated', 'hands',
    'required', 'preferred', 'must', 'desire', 'desired', 'ideal',
    'responsibilities', 'responsibility', 'requirements', 'requirement',
    'role', 'roles', 'position', 'job', 'company', 'organization', 'team',
    'looking', 'seeking', 'candidate', 'candidates',
    'application', 'apply', 'email', 'description', 'summary', 'overview',
    'knowledge', 'understanding', 'expertise',
    'new', 'using', 'used', 'use', 'uses',
    # generic action verbs (non-technical)
    'create', 'creating', 'created', 'creation',
    'build', 'building', 'built',
    'develop', 'developing', 'developed', 'development',
    'manage', 'managing', 'managed', 'management',
    'deliver', 'delivering', 'delivered', 'delivery',
    'support', 'supporting', 'supported',
    'maintain', 'maintaining', 'maintained',
    'provide', 'providing', 'provided',
    'collaborate', 'collaborating', 'collaborated',
    'participate', 'participating', 'participated',
    'update', 'updating', 'updated',
    'enhance', 'enhancing', 'enhanced',
    'implement', 'implementing', 'implemented',
    'design', 'designing', 'designed',
    'contribute', 'contributing', 'contributed',
    'integrate', 'integrating', 'integrated',
    'optimize', 'optimizing', 'optimized',
    'achieve', 'achieving', 'achieved', 'achieving',
    'reduce', 'reducing', 'reduced',
    'enable', 'enabling', 'enabled',
    'produce', 'producing', 'produced',
    'convert', 'converting', 'converted',
    'translate', 'translating', 'translated',
    'configure', 'configuring', 'configured',
    'troubleshoot', 'troubleshooting',
    # generic adjectives / nouns that aren't actionable
    'high', 'low', 'large', 'small', 'end', 'multi', 'cross',
    'key', 'core', 'based', 'related', 'driven', 'grade', 'native',
    'daily', 'basic', 'soft', 'hard', 'time', 'following',
    'quality', 'solutions', 'solution', 'products', 'product',
    'clients', 'client', 'stakeholders', 'stakeholder',
    'effectively', 'efficiently', 'across', 'various',
    'robust', 'resilient', 'scalable', 'production',
    'potential', 'functional', 'technical', 'practice', 'practices',
    'impact', 'foundation', 'profile', 'cases', 'case',
    'sits', 'intersection', 'requiring', 'thinking',
    'flow', 'diagrams', 'specifications',
    'engineer', 'associate', 'consultant',
    'enterprise', 'distributed',
    # more generic terms from JD analysis
    'leveraging', 'adhering', 'contextual', 'deployable',
    'depth', 'domains', 'sits', 'translate',
    'responses', 'actions', 'error', 'errors',
    'intelligent', 'decision', 'efficient', 'efficiency',
    'systems', 'system', 'services', 'service',
    'frameworks', 'framework', 'tools', 'tool',
    'models', 'model', 'page', 'pages', 'documents', 'document',
    'environments', 'environment',
    'structured', 'modular', 'automated',
    'setup', 'governance', 'rightsizing', 'resource',
    'accuracy', 'matching', 'prediction', 'modeling',
    'taxonomy', 'salary', 'communication',
    'cost', 'documentation', 'competencies',
    'professional', 'analytical', 'analyzer',
    # words only meaningful as part of bigrams, not standalone
    'language', 'augmented', 'artificial', 'intelligence',
    'machine', 'learning', 'retrieval', 'generation',
    'engineered',
    'principles', 'templates',
    'workflows', 'processing',
    # misc
    'etc', 'per', 'via',
}


# ────────────────────────────────────────────────────────────────────
# TECHNICAL TERMS – these are the keywords that actually matter
# ────────────────────────────────────────────────────────────────────
TECHNICAL_TERMS: Set[str] = {
    # programming languages
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
    'ruby', 'php', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl',
    'pl/sql', 'sql', 'nosql', 'html', 'css', 'bash', 'shell',
    # frameworks / libraries
    'react', 'angular', 'vue', 'nextjs', 'node', 'express', 'django', 'flask',
    'spring', 'fastapi', 'rails', 'laravel', '.net', 'tensorflow', 'pytorch',
    'keras', 'scikit', 'scikit-learn', 'pandas', 'numpy', 'opencv',
    'langchain', 'autogen', 'llamaindex',
    # databases
    'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch', 'dynamodb',
    'oracle', 'cassandra', 'neo4j', 'sqlite', 'rds',
    'faiss', 'chromadb', 'pinecone', 'weaviate',
    # cloud / infra
    'aws', 'azure', 'gcp', 'ec2', 's3', 'lambda', 'iam', 'vpc',
    'cloudformation', 'terraform', 'ansible', 'pulumi', 'arm',
    'docker', 'kubernetes', 'k8s', 'helm', 'nginx',
    'jenkins', 'github', 'gitlab', 'bitbucket',
    # AI / ML / NLP
    'llm', 'llms', 'rag', 'embeddings', 'transformers',
    'openai', 'claude', 'gemini', 'gpt',
    'bert', 'lstm', 'cnn', 'gan',
    'nlp', 'nlu', 'ner',
    # concepts / methodologies
    'agile', 'scrum', 'kanban', 'devops', 'mlops',
    'ci/cd', 'cicd', 'git', 'linux', 'unix',
    'microservices', 'serverless', 'api', 'apis', 'rest', 'restful', 'graphql', 'grpc',
    'containerization', 'orchestration',
    'asynchronous', 'async', 'concurrency', 'multithreading',
    'logging', 'monitoring', 'analytics', 'observability',
    'prometheus', 'grafana', 'datadog', 'splunk', 'spotfire',
    # data / ML concepts
    'etl', 'pipeline', 'pipelines', 'warehouse', 'datalake',
    'feature', 'classification', 'regression',
    'clustering', 'reinforcement',
    # security / networking
    'oauth', 'jwt', 'ssl', 'tls', 'https', 'encryption',
    'firewall', 'vpn', 'dns', 'cdn',
    # certifications keywords
    'certified', 'practitioner', 'certification', 'certifications',
    'foundations', 'ibm',
    # prompt / semantic
    'prompt', 'semantic', 'vector',
}


# Technical bigrams (whitelist only)
IMPORTANT_BIGRAMS: Set[str] = {
    'machine learning', 'deep learning', 'data science', 'data analysis',
    'data engineering', 'data pipeline', 'data warehouse', 'data modeling',
    'project management', 'software development', 'web development',
    'full stack', 'front end', 'back end', 'cloud computing',
    'cloud infrastructure', 'cloud transformation',
    'artificial intelligence', 'natural language', 'computer vision',
    'big data', 'api development', 'system design',
    'database management', 'version control',
    'continuous integration', 'continuous deployment',
    'test driven', 'object oriented', 'functional programming',
    'agile methodology', 'scrum master', 'product management',
    'ci cd', 'rest api', 'graphql api', 'microservices architecture',
    'infrastructure provisioning', 'performance tuning',
    'prompt engineering', 'vector databases', 'semantic search',
    'retrieval augmented', 'augmented generation',
    'language models', 'agent orchestration',
    'scikit learn',
}

# Technical trigrams
IMPORTANT_TRIGRAMS: Set[str] = {
    'machine learning engineer', 'full stack developer',
    'senior software engineer', 'natural language processing',
    'test driven development', 'object oriented programming',
    'amazon web services', 'google cloud platform',
    'retrieval augmented generation', 'large language models',
    'large language model',
}


def extract_keywords(text: str, min_length: int = 3, use_ngrams: bool = True) -> List[str]:
    """
    Extract meaningful keywords from text with N-gram support.
    """
    # Clean text: keep letters, digits, +, #, /
    text_clean = re.sub(r'[^a-zA-Z0-9\s+#/]', ' ', text.lower())

    words = text_clean.split()

    # Filter single words
    single_keywords = []
    for word in words:
        # Strip trailing/leading junk, skip too-short / stop / numeric
        word = word.strip('.')
        if (
            len(word) >= min_length
            and word not in STOP_WORDS
            and not word.isdigit()
            and re.search(r'[a-zA-Z]', word)
        ):
            single_keywords.append(word)

    keywords = single_keywords.copy()

    # N-grams: only recognised technical phrases
    if use_ngrams and len(words) >= 2:
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i + 1]}"
            if bigram in IMPORTANT_BIGRAMS:
                keywords.append(bigram)

        for i in range(len(words) - 2):
            trigram = f"{words[i]} {words[i + 1]} {words[i + 2]}"
            if trigram in IMPORTANT_TRIGRAMS:
                keywords.append(trigram)

    return keywords


def calculate_tfidf_weights(keywords: List[str], corpus_keywords: List[str] = None) -> Dict[str, float]:
    """
    Calculate TF-IDF-like weights for keywords.
    Higher weight = more important/rare term.
    """
    tf = Counter(keywords)
    total_terms = len(keywords)

    weights = {}
    for keyword, count in tf.items():
        base_weight = count / total_terms if total_terms > 0 else 0
        keyword_lower = keyword.lower()

        if keyword_lower in TECHNICAL_TERMS or any(t in keyword_lower for t in TECHNICAL_TERMS if len(t) > 3):
            weights[keyword] = base_weight * 3.0
        elif len(keyword.split()) >= 2:
            weights[keyword] = base_weight * 2.0
        else:
            weights[keyword] = base_weight * 1.0

    # Normalize to 0-1
    if weights:
        max_weight = max(weights.values())
        if max_weight > 0:
            weights = {k: v / max_weight for k, v in weights.items()}

    return weights


def calculate_keyword_overlap(resume_keywords: List[str], jd_keywords: List[str]) -> float:
    """
    Calculate keyword overlap between resume and JD using TF-IDF weighting.
    """
    if not jd_keywords:
        return 0.0

    jd_weights = calculate_tfidf_weights(jd_keywords)
    resume_set = set(kw.lower() for kw in resume_keywords)

    matched_weight = 0.0
    total_weight = 0.0

    for keyword in jd_keywords:
        keyword_lower = keyword.lower()
        weight = jd_weights.get(keyword, 0.1)
        total_weight += weight

        if keyword_lower in resume_set:
            matched_weight += weight
        else:
            for resume_kw in resume_set:
                if keyword_lower in resume_kw or resume_kw in keyword_lower:
                    matched_weight += weight * 0.7
                    break

    if total_weight == 0:
        return 0.0

    raw_score = (matched_weight / total_weight) * 100
    return min(100, raw_score * 1.2)


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def get_missing_keywords(resume_keywords: List[str], jd_keywords: List[str]) -> Dict[str, List[str]]:
    """
    Identify keywords present in JD but missing from resume.
    Categorises by actual importance: technical terms are critical,
    recognised phrases are important, everything else is optional.
    """
    if not jd_keywords:
        return {'critical': [], 'important': [], 'optional': []}

    resume_set = set(kw.lower() for kw in resume_keywords)

    # Also build a flat set of individual words from all resume keywords
    # This helps match multi-word JD phrases when the resume has the
    # individual words (e.g. JD: "microservices architecture" vs resume
    # having "microservices" and "architecture" separately)
    resume_words = set()
    for kw in resume_set:
        for word in kw.split():
            if len(word) >= 3:
                resume_words.add(word)

    def _is_matched(kw_lower: str) -> bool:
        """Check if a keyword is present in the resume via multiple strategies."""
        # Exact match
        if kw_lower in resume_set:
            return True
        # Substring match (e.g. "sql" in "postgresql")
        if any(kw_lower in rkw or rkw in kw_lower for rkw in resume_set):
            return True
        # For multi-word phrases: check if ALL words exist in resume
        words = kw_lower.split()
        if len(words) >= 2 and all(w in resume_words for w in words if len(w) >= 3):
            return True
        # Stem match: "llms" matches "llm"
        stem = kw_lower.rstrip('s')
        if stem != kw_lower and (stem in resume_set or any(stem in rkw for rkw in resume_set)):
            return True
        return False

    # Deduplicate and find truly missing keywords
    seen: set[str] = set()
    missing: list[str] = []
    for keyword in jd_keywords:
        kw_lower = keyword.lower()
        if kw_lower in seen:
            continue
        seen.add(kw_lower)
        if not _is_matched(kw_lower):
            missing.append(keyword)

    if not missing:
        return {'critical': [], 'important': [], 'optional': []}

    # Categorise based on whether the term is a recognised technical keyword
    critical: list[str] = []
    important: list[str] = []
    optional: list[str] = []

    for kw in missing:
        kw_lower = kw.lower()
        kw_stem = kw_lower.rstrip('s')
        is_tech = (
            kw_lower in TECHNICAL_TERMS
            or kw_stem in TECHNICAL_TERMS
            or (kw_lower + 's') in TECHNICAL_TERMS
            or any(t in kw_lower for t in TECHNICAL_TERMS if len(t) >= 3)
        )
        is_phrase = len(kw.split()) >= 2

        if is_tech:
            critical.append(kw)
        elif is_phrase:
            important.append(kw)
        else:
            optional.append(kw)

    # Cap each category for readability
    return {
        'critical': critical[:10],
        'important': important[:10],
        'optional': optional[:15],
    }
