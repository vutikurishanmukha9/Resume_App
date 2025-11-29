from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os
from werkzeug.utils import secure_filename
import PyPDF2
import logging
import pickle
from contextlib import contextmanager
from typing import Tuple, List, Dict, Any
import traceback
import re
from collections import Counter
import json
from datetime import datetime

# -------------------- CONFIGURATION --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt'}

@app.route("/health")
def health():
    return "ok", 200

# Constants
MAX_TEXT_LENGTH = 5000
MAX_PDF_PAGES = 5
TOP_MATCHES = 3
EMBEDDING_CACHE_FILE = "job_embeddings_cache.pkl"
MIN_TEXT_LENGTH = 50

# Matching weights for different components
MATCHING_WEIGHTS = {
    'semantic': 0.40,      # Semantic similarity
    'keyword': 0.30,       # Keyword overlap
    'skills': 0.20,        # Skills matching
    'context': 0.10        # Contextual similarity
}

# Analytics
ANALYTICS_FILE = "analytics.json"

# Rate limiting (will be configured with Flask-Limiter)
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"],
        storage_uri="memory://"
    )
    logger.info("Rate limiting enabled")
except ImportError:
    logger.warning("Flask-Limiter not installed. Rate limiting disabled.")
    limiter = None


# -------------------- ANALYTICS TRACKING --------------------

def track_analysis(analysis_type: str, data: Dict[str, Any]) -> None:
    """Track analysis for analytics purposes"""
    try:
        # Load existing analytics
        if os.path.exists(ANALYTICS_FILE):
            with open(ANALYTICS_FILE, 'r') as f:
                analytics = json.load(f)
        else:
            analytics = {
                'analyses': [],
                'summary': {
                    'total_uploads': 0,
                    'total_matches': 0,
                    'avg_match_score': 0.0,
                    'last_updated': None
                }
            }
        
        # Add new analysis
        analysis_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': analysis_type,
            'data': data
        }
        analytics['analyses'].append(analysis_entry)
        
        # Update summary
        if analysis_type == 'upload':
            analytics['summary']['total_uploads'] += 1
        elif analysis_type == 'jd_match':
            analytics['summary']['total_matches'] += 1
            # Update average match score
            match_scores = [a['data'].get('match_percentage', 0) 
                          for a in analytics['analyses'] 
                          if a['type'] == 'jd_match']
            if match_scores:
                analytics['summary']['avg_match_score'] = sum(match_scores) / len(match_scores)
        
        analytics['summary']['last_updated'] = datetime.now().isoformat()
        
        # Save analytics (keep last 1000 entries)
        analytics['analyses'] = analytics['analyses'][-1000:]
        
        with open(ANALYTICS_FILE, 'w') as f:
            json.dump(analytics, f, indent=2)
            
    except Exception as e:
        logger.warning(f"Failed to track analytics: {e}")


# -------------------- MODEL MANAGER --------------------
class ModelManager:
    """Centralized model management with caching and error handling"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.resume_classifier = None
        self.salary_model = None
        self.job_df = None
        self.embed_model = None
        self.job_embeddings = None
        self._models_loaded = False
        self._initialized = True

    def load_models(self):
        """Load all ML models with comprehensive error handling"""
        if self._models_loaded:
            logger.info("Models already loaded, skipping...")
            return

        try:
            logger.info("Loading models...")

            # Load classifier
            if not os.path.exists("job_classifier.pkl"):
                raise FileNotFoundError("job_classifier.pkl not found")
            self.resume_classifier = joblib.load("job_classifier.pkl")
            logger.info("Resume classifier loaded")

            # Load salary predictor
            if not os.path.exists("salary_predictor.pkl"):
                raise FileNotFoundError("salary_predictor.pkl not found")
            self.salary_model = joblib.load("salary_predictor.pkl")
            logger.info("Salary predictor loaded")

            # Load job dataset
            if not os.path.exists("job_title_des.csv"):
                raise FileNotFoundError("job_title_des.csv not found")
            self.job_df = pd.read_csv("job_title_des.csv")
            
            # Validate dataset columns
            required_columns = ['Job Description', 'Job Title']
            if not all(col in self.job_df.columns for col in required_columns):
                raise ValueError(f"Dataset must contain {required_columns} columns")
            
            # Remove any rows with missing critical data
            self.job_df = self.job_df.dropna(subset=required_columns)
            logger.info(f"Job dataset loaded with {len(self.job_df)} entries")

            # Load embedding model
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embed_model.max_seq_length = 256
            logger.info("Sentence Transformer model loaded")

            # Precompute embeddings
            self.precompute_job_embeddings()

            self._models_loaded = True
            logger.info("All models successfully initialized")

        except FileNotFoundError as e:
            logger.error(f"Required file missing: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to initialize models: {str(e)}")

    def precompute_job_embeddings(self):
        """Precompute embeddings for job descriptions with validation"""
        try:
            # Try to load cached embeddings
            if os.path.exists(EMBEDDING_CACHE_FILE):
                try:
                    with open(EMBEDDING_CACHE_FILE, 'rb') as f:
                        self.job_embeddings = pickle.load(f)
                    
                    # Validate cache matches current dataset
                    if len(self.job_embeddings) == len(self.job_df):
                        logger.info("Loaded cached job embeddings")
                        return
                    else:
                        logger.warning("Cache size mismatch, recomputing embeddings...")
                except Exception as e:
                    logger.warning(f"Cache load failed: {e}, recomputing embeddings...")

            # Compute new embeddings
            job_descriptions = self.job_df['Job Description'].fillna('').tolist()
            
            if not job_descriptions:
                raise ValueError("No job descriptions found in dataset")
            
            logger.info(f"Computing embeddings for {len(job_descriptions)} job descriptions...")
            self.job_embeddings = self.embed_model.encode(
                job_descriptions, 
                convert_to_tensor=True, 
                show_progress_bar=True,
                batch_size=32
            )
            
            # Cache the embeddings
            with open(EMBEDDING_CACHE_FILE, 'wb') as f:
                pickle.dump(self.job_embeddings, f)
            logger.info("Job embeddings computed and cached")
            
        except Exception as e:
            logger.error(f"Failed to compute embeddings: {e}")
            raise

    def is_loaded(self):
        """Check if models are loaded"""
        return self._models_loaded


model_manager = ModelManager()


# -------------------- TEXT PROCESSING UTILITIES --------------------

def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """Extract meaningful keywords from text"""
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s+#]', ' ', text.lower())
    
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
        'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'or', 'and'
    }
    
    # Extract words
    words = text.split()
    keywords = [
        word for word in words 
        if len(word) >= min_length and word not in stop_words
    ]
    
    return keywords


def extract_skills(text: str) -> Dict[str, set]:
    """
    Extract technical skills and technologies from text using comprehensive taxonomy.
    Returns categorized skills with normalized names.
    """
    text_lower = text.lower()
    
    # Load skills taxonomy
    try:
        import json
        taxonomy_path = os.path.join(os.path.dirname(__file__), 'skills_taxonomy.json')
        with open(taxonomy_path, 'r', encoding='utf-8') as f:
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


def calculate_keyword_overlap(resume_keywords: List[str], jd_keywords: List[str]) -> float:
    """Calculate keyword overlap between resume and JD"""
    if not jd_keywords:
        return 0.0
    
    resume_counter = Counter(resume_keywords)
    jd_counter = Counter(jd_keywords)
    
    # Calculate intersection
    common_keywords = set(resume_counter.keys()) & set(jd_counter.keys())
    
    if not common_keywords:
        return 0.0
    
    # Weight by frequency
    overlap_score = sum(
        min(resume_counter[kw], jd_counter[kw]) for kw in common_keywords
    )
    total_jd_keywords = sum(jd_counter.values())
    
    return min(100, (overlap_score / total_jd_keywords) * 100)


def calculate_skills_match(resume_skills: set, jd_skills: set) -> float:
    """Calculate skills matching percentage"""
    if not jd_skills:
        return 100.0  # If no specific skills in JD, give full score
    
    matching_skills = resume_skills & jd_skills
    return (len(matching_skills) / len(jd_skills)) * 100


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


# -------------------- CONTEXT & HELPERS --------------------
@contextmanager
def temporary_file(file_path):
    """Temporary file cleanup context manager"""
    try:
        yield file_path
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {file_path}: {e}")


def allowed_file(filename: str) -> bool:
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file with error handling"""
    try:
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = min(len(reader.pages), MAX_PDF_PAGES)
            
            for i in range(num_pages):
                try:
                    content = reader.pages[i].extract_text()
                    if content:
                        text += content + "\n"
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {i+1}: {e}")
                    continue
        
        extracted = text.strip()[:MAX_TEXT_LENGTH]
        
        if len(extracted) < MIN_TEXT_LENGTH:
            raise ValueError("Insufficient text extracted from PDF. Please ensure the PDF contains readable text.")
        
        return extracted
        
    except PyPDF2.errors.PdfReadError as e:
        raise ValueError(f"Invalid or corrupted PDF file: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to process PDF: {str(e)}")


def extract_text_from_file(file_path: str, filename: str) -> str:
    """Extract text from PDF or TXT file with validation"""
    ext = filename.rsplit('.', 1)[-1].lower()
    
    try:
        if ext == 'pdf':
            return extract_text_from_pdf(file_path)
        elif ext == 'txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()[:MAX_TEXT_LENGTH]
                
            if len(content) < MIN_TEXT_LENGTH:
                raise ValueError("Text file is too short. Please provide a resume with at least 50 characters.")
            
            return content
        else:
            raise ValueError("Unsupported file type")
    except UnicodeDecodeError:
        raise ValueError("Failed to read text file. Please ensure it's a valid UTF-8 encoded text file.")


# -------------------- FEATURE EXTRACTION FOR SALARY PREDICTION --------------------

def extract_years_of_experience(text: str) -> float:
    """
    Extract years of experience from resume text.
    Looks for patterns like "X years", "X+ years", date ranges, etc.
    """
    text_lower = text.lower()
    years = []
    
    # Pattern 1: "X years of experience" or "X+ years"
    pattern1 = r'(\d+)\+?\s*years?\s*(?:of)?\s*(?:experience|exp)'
    matches1 = re.findall(pattern1, text_lower)
    years.extend([int(m) for m in matches1])
    
    # Pattern 2: Date ranges (e.g., "2019 - 2023", "Jan 2020 - Present")
    # Extract years from dates
    year_pattern = r'\b(19|20)\d{2}\b'
    year_matches = re.findall(year_pattern, text)
    
    if len(year_matches) >= 2:
        # Calculate experience from date ranges
        years_found = [int(y) for y in year_matches]
        years_found.sort()
        # Assume most recent experience
        if years_found:
            current_year = 2024  # Update this dynamically if needed
            max_year = max(years_found)
            min_year = min(years_found)
            calculated_exp = max_year - min_year
            if calculated_exp > 0 and calculated_exp < 50:  # Sanity check
                years.append(calculated_exp)
    
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


# -------------------- CORE ANALYSIS --------------------
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
            except Exception as e:
                logger.warning(f"Failed to process match at index {idx}: {e}")
                continue

        # Extract features for salary prediction
        features = extract_resume_features(resume_text)
        
        # Create feature vector for salary model
        # Note: The model was trained with a single feature, so we'll use a weighted combination
        # In production, retrain the model with all these features
        feature_vector = np.array([[
            features['years_experience'] * 2 +  # Weight experience heavily
            features['education_level'] * 3 +   # Weight education
            features['seniority_level'] * 2.5 + # Weight seniority
            features['skills_count'] * 0.5      # Weight skills count
        ]])
        
        # Predict salary using extracted features
        predicted_salary = float(model_manager.salary_model.predict(feature_vector)[0])
        
        # Prepare salary details for response
        salary_details = {
            'features': {
                'years_experience': features['years_experience'],
                'education_level': features['education_level'],
                'seniority_level': features['seniority_level'],
                'skills_count': features['skills_count']
            },
            'confidence': round(features['completeness'], 2),
            'note': 'Salary prediction based on extracted resume features. Confidence indicates feature completeness.'
        }
        
        return predicted_job, matches, predicted_salary, salary_details
        
    except Exception as e:
        logger.error(f"Resume analysis failed: {e}")
        raise ValueError(f"Failed to analyze resume: {str(e)}")


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
        semantic_score = util.cos_sim(resume_embedding, jd_embedding).item() * 100
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
            
            if jd_cat_skills:  # Only include categories that JD requires
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
        # Compare sentence-level embeddings for better context
        resume_sentences = split_into_sentences(resume_text)[:10]  # Top 10 sentences
        jd_sentences = split_into_sentences(jd_text)[:10]
        
        if resume_sentences and jd_sentences:
            resume_sent_embeds = model_manager.embed_model.encode(
                resume_sentences, 
                convert_to_tensor=True
            )
            jd_sent_embeds = model_manager.embed_model.encode(
                jd_sentences, 
                convert_to_tensor=True
            )
            
            # Calculate max similarity for each JD sentence
            similarities = util.cos_sim(jd_sent_embeds, resume_sent_embeds)
            max_sims = similarities.max(dim=1).values
            contextual_score = max_sims.mean().item() * 100
        else:
            contextual_score = semantic_score  # Fallback to semantic score
        
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
        
    except Exception as e:
        logger.error(f"JD matching failed: {e}")
        raise ValueError(f"Failed to calculate match: {str(e)}")


# -------------------- ROUTES --------------------
@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
@limiter.limit("10 per minute") if limiter else lambda f: f
def upload_resume():
    """Handle resume upload and analysis"""
    try:
        # Check if models are loaded
        if not model_manager.is_loaded():
            return jsonify({'error': 'System is still initializing. Please try again.'}), 503

        # Validate file presence
        if 'resume' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['resume']
        
        if not file or file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PDF and TXT files are allowed.'}), 400

        # Secure filename
        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({'error': 'Invalid filename'}), 400
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Process file
        with temporary_file(file_path):
            file.save(file_path)
            
            # Extract text
            resume_text = extract_text_from_file(file_path, filename)
            
            # Analyze resume
            predicted_job, matches, salary, salary_details = analyze_resume(resume_text)
            
            # Track analytics
            track_analysis('upload', {
                'predicted_job': predicted_job,
                'salary': int(salary),
                'confidence': salary_details['confidence']
            })

            return jsonify({
                'success': True,
                'predicted_job': predicted_job,
                'matches': [{'title': t, 'score': f"{s:.3f}"} for t, s in matches],
                'salary': f"₹{int(salary):,}",
                'salary_details': salary_details
            }), 200

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Upload error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500


@app.route('/match_jd_resume', methods=['POST'])
@limiter.limit("10 per minute") if limiter else lambda f: f
def match_jd_resume():
    """Calculate resume-JD match percentage with detailed breakdown"""
    try:
        # Check if models are loaded
        if not model_manager.is_loaded():
            return jsonify({'error': 'System is still initializing. Please try again.'}), 503

        # Get inputs
        jd_text = request.form.get('jd_text', '').strip()
        
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file uploaded'}), 400
        
        resume_file = request.files['resume']

        if not jd_text:
            return jsonify({'error': 'Please provide a job description'}), 400
        
        if not resume_file or resume_file.filename == '':
            return jsonify({'error': 'Please upload a resume file'}), 400
        
        if not allowed_file(resume_file.filename):
            return jsonify({'error': 'Invalid file type. Only PDF and TXT files are allowed.'}), 400

        # Process file
        filename = secure_filename(resume_file.filename)
        if not filename:
            return jsonify({'error': 'Invalid filename'}), 400
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        with temporary_file(file_path):
            resume_file.save(file_path)
            
            # Extract text
            resume_text = extract_text_from_file(file_path, filename)
            
            # Calculate match with detailed breakdown
            match_percentage, detailed_results = calculate_jd_resume_match(resume_text, jd_text)
            
            # Track analytics
            track_analysis('jd_match', {
                'match_percentage': match_percentage,
                'component_scores': detailed_results.get('component_scores', {})
            })

            return jsonify({
                'success': True,
                'match_percentage': match_percentage,
                'component_scores': detailed_results.get('component_scores', {}),
                'missing_keywords': detailed_results.get('missing_keywords', {}),
                'keyword_suggestions': detailed_results.get('keyword_suggestions', []),
                'skills_breakdown': detailed_results.get('skills_breakdown', {}),
                'message': f"The resume matches {match_percentage}% with the job description"
            }), 200

    except ValueError as e:
        logger.warning(f"Validation error in JD match: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"JD Match Error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to calculate match percentage. Please try again.'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': model_manager.is_loaded()
    }), 200


# -------------------- ERROR HANDLERS --------------------
@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File size exceeds 16MB limit'}), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


# -------------------- MAIN ENTRY --------------------
if __name__ == '__main__':
    # Create upload folder
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    try:
        logger.info("Initializing AI Resume Analyzer...")
        model_manager.load_models()
        logger.info("System ready!")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        logger.error(traceback.format_exc())
        exit(1)

    app.run(debug=True, host='0.0.0.0', port=5000)

    

'''from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os
from werkzeug.utils import secure_filename
import PyPDF2
import logging
import pickle
import hashlib
import time
from contextlib import contextmanager
from typing import Tuple, List, Dict, Any, Optional, Set
import traceback
import re
from collections import Counter
from functools import lru_cache

# -------------------- CONFIGURATION --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt'}

# Constants
MAX_TEXT_LENGTH = 5000
MAX_PDF_PAGES = 5
TOP_MATCHES = 3
EMBEDDING_CACHE_FILE = "job_embeddings_cache.pkl"
EMBEDDING_HASH_FILE = "job_embeddings_hash.txt"
MIN_TEXT_LENGTH = 50
MIN_KEYWORD_LENGTH = 3
MIN_SENTENCE_LENGTH = 20
EMBEDDING_MAX_SEQ_LENGTH = 256
EMBEDDING_BATCH_SIZE = 32

# Matching weights for different components
MATCHING_WEIGHTS = {
    'semantic': 0.40,      # Semantic similarity
    'keyword': 0.30,       # Keyword overlap
    'skills': 0.20,        # Skills matching
    'context': 0.10        # Contextual similarity
}

# Compile regex patterns once at module level for performance
SKILL_PATTERNS = [
    re.compile(r'\b(python|java|javascript|typescript|c\+\+|c#|ruby|php|swift|kotlin|go|rust|scala|r|matlab)\b', re.IGNORECASE),
    re.compile(r'\b(html|css|react|angular|vue|node\.?js|express|django|flask|spring|asp\.net)\b', re.IGNORECASE),
    re.compile(r'\b(sql|mysql|postgresql|mongodb|redis|oracle|sqlite|cassandra|dynamodb)\b', re.IGNORECASE),
    re.compile(r'\b(aws|azure|gcp|docker|kubernetes|jenkins|terraform|ansible|git|github|gitlab)\b', re.IGNORECASE),
    re.compile(r'\b(machine learning|deep learning|tensorflow|pytorch|scikit-learn|pandas|numpy|matplotlib|nlp|computer vision)\b', re.IGNORECASE),
    re.compile(r'\b(rest|api|microservices|agile|scrum|ci/cd|linux|unix|windows|bash|powershell)\b', re.IGNORECASE),
    re.compile(r'\b(leadership|communication|problem solving|team work|analytical|project management)\b', re.IGNORECASE)
]

# Stop words for keyword extraction
STOP_WORDS = frozenset({
    'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was', 'were',
    'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
    'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'who', 'when',
    'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'just', 'but', 'for', 'with', 'about', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
    'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'or', 'and'
})

# -------------------- MODEL LOADER --------------------
_embed_model_cache = {}

def load_embed_model(quick_mode: bool = True) -> SentenceTransformer:
    """
    Load and cache embedding model dynamically.
    quick_mode=True → all-MiniLM-L12-v2 (fast and balanced)
    quick_mode=False → all-mpnet-base-v2 (high accuracy)
    """
    cache_key = f"quick_{quick_mode}"
    
    if cache_key in _embed_model_cache:
        return _embed_model_cache[cache_key]
    
    if quick_mode:
        logger.info(" Loading embedding model: all-MiniLM-L12-v2 (Fast Mode)")
        model = SentenceTransformer('all-MiniLM-L12-v2')
    else:
        logger.info(" Loading embedding model: all-mpnet-base-v2 (Accuracy Mode)")
        model = SentenceTransformer('all-mpnet-base-v2')

    model.max_seq_length = EMBEDDING_MAX_SEQ_LENGTH
    _embed_model_cache[cache_key] = model
    return model


# -------------------- MODEL MANAGER --------------------
class ModelManager:
    """Centralized model management with caching and error handling"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.resume_classifier: Optional[Any] = None
        self.salary_model: Optional[Any] = None
        self.job_df: Optional[pd.DataFrame] = None
        self.embed_model: Optional[SentenceTransformer] = None
        self.job_embeddings: Optional[Any] = None
        self._models_loaded = False
        self._initialized = True

    def _compute_dataset_hash(self) -> str:
        """Compute hash of job dataset to detect changes"""
        try:
            content = self.job_df[['Job Description', 'Job Title']].to_string()
            return hashlib.md5(content.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Hash computation failed: {e}")
            return ""

    def _load_cached_hash(self) -> Optional[str]:
        """Load previously saved dataset hash"""
        try:
            if os.path.exists(EMBEDDING_HASH_FILE):
                with open(EMBEDDING_HASH_FILE, 'r') as f:
                    return f.read().strip()
        except Exception as e:
            logger.warning(f"Failed to load hash file: {e}")
        return None

    def _save_hash(self, hash_value: str) -> None:
        """Save dataset hash to file"""
        try:
            with open(EMBEDDING_HASH_FILE, 'w') as f:
                f.write(hash_value)
        except Exception as e:
            logger.warning(f"Failed to save hash: {e}")

    def load_models(self, quick_mode: bool = True) -> None:
        """Load all ML models with comprehensive error handling"""
        if self._models_loaded:
            logger.info("Models already loaded, skipping...")
            return

        start_time = time.time()
        
        try:
            logger.info(" Loading models...")

            # Load classifier
            if not os.path.exists("job_classifier.pkl"):
                raise FileNotFoundError("job_classifier.pkl not found")
            self.resume_classifier = joblib.load("job_classifier.pkl")
            logger.info("Resume classifier loaded")

            # Load salary predictor
            if not os.path.exists("salary_predictor.pkl"):
                raise FileNotFoundError("salary_predictor.pkl not found")
            self.salary_model = joblib.load("salary_predictor.pkl")
            logger.info("Salary predictor loaded")

            # Load job dataset
            if not os.path.exists("job_title_des.csv"):
                raise FileNotFoundError("job_title_des.csv not found")
            self.job_df = pd.read_csv("job_title_des.csv")

            # Validate dataset columns
            required_columns = ['Job Description', 'Job Title']
            missing_columns = [col for col in required_columns if col not in self.job_df.columns]
            if missing_columns:
                raise ValueError(f"Dataset missing required columns: {missing_columns}")

            # Remove any rows with missing critical data
            initial_rows = len(self.job_df)
            self.job_df = self.job_df.dropna(subset=required_columns)
            removed_rows = initial_rows - len(self.job_df)
            
            if removed_rows > 0:
                logger.info(f"Removed {removed_rows} rows with missing data")
            
            if len(self.job_df) == 0:
                raise ValueError("No valid job entries found in dataset")
                
            logger.info(f"Job dataset loaded with {len(self.job_df)} entries")

            # Load embedding model (Dynamic choice)
            self.embed_model = load_embed_model(quick_mode)
            logger.info("Sentence Transformer model loaded")

            # Precompute embeddings
            self.precompute_job_embeddings()

            self._models_loaded = True
            elapsed = time.time() - start_time
            logger.info(f"All models successfully initialized in {elapsed:.2f}s")

        except FileNotFoundError as e:
            logger.error(f"Required file missing: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
        except ValueError as e:
            logger.error(f"Data validation error: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to initialize models: {str(e)}")

    def precompute_job_embeddings(self) -> None:
        """Precompute embeddings for job descriptions with validation"""
        try:
            current_hash = self._compute_dataset_hash()
            cached_hash = self._load_cached_hash()
            
            # Check if we can use cached embeddings
            if os.path.exists(EMBEDDING_CACHE_FILE) and current_hash == cached_hash:
                try:
                    with open(EMBEDDING_CACHE_FILE, 'rb') as f:
                        self.job_embeddings = pickle.load(f)
                    if len(self.job_embeddings) == len(self.job_df):
                        logger.info("Loaded cached job embeddings (validated)")
                        return
                    else:
                        logger.warning("Cache size mismatch, recomputing embeddings...")
                except Exception as e:
                    logger.warning(f"Cache load failed: {e}, recomputing embeddings...")
            else:
                if cached_hash != current_hash:
                    logger.info("Dataset changed detected, recomputing embeddings...")

            job_descriptions = self.job_df['Job Description'].fillna('').tolist()

            if not job_descriptions:
                raise ValueError("No job descriptions found in dataset")

            logger.info(f"Computing embeddings for {len(job_descriptions)} job descriptions...")
            self.job_embeddings = self.embed_model.encode(
                job_descriptions,
                convert_to_tensor=True,
                show_progress_bar=True,
                batch_size=EMBEDDING_BATCH_SIZE
            )

            # Save embeddings and hash
            with open(EMBEDDING_CACHE_FILE, 'wb') as f:
                pickle.dump(self.job_embeddings, f)
            self._save_hash(current_hash)
            
            logger.info("Job embeddings computed and cached")

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to compute embeddings: {e}")
            logger.error(traceback.format_exc())
            raise

    def is_loaded(self) -> bool:
        """Check if models are loaded"""
        return self._models_loaded


model_manager = ModelManager()

# -------------------- TEXT PROCESSING UTILITIES --------------------
def extract_keywords(text: str, min_length: int = MIN_KEYWORD_LENGTH) -> List[str]:
    """Extract meaningful keywords from text"""
    if not text or not isinstance(text, str):
        return []
    
    text = re.sub(r'[^a-zA-Z0-9\s+#]', ' ', text.lower())
    words = text.split()
    keywords = [word for word in words if len(word) >= min_length and word not in STOP_WORDS]
    return keywords


def extract_skills(text: str) -> Set[str]:
    """Extract technical skills and technologies from text"""
    if not text or not isinstance(text, str):
        return set()
    
    text_lower = text.lower()
    skills = set()
    
    for pattern in SKILL_PATTERNS:
        matches = pattern.findall(text_lower)
        skills.update(matches)
    
    return skills


def calculate_keyword_overlap(resume_keywords: List[str], jd_keywords: List[str]) -> float:
    """Calculate keyword overlap between resume and JD"""
    if not jd_keywords or not isinstance(jd_keywords, list):
        return 0.0
    
    if not resume_keywords or not isinstance(resume_keywords, list):
        return 0.0
    
    resume_counter = Counter(resume_keywords)
    jd_counter = Counter(jd_keywords)
    common_keywords = set(resume_counter) & set(jd_counter)
    
    if not common_keywords:
        return 0.0
    
    overlap_score = sum(min(resume_counter[kw], jd_counter[kw]) for kw in common_keywords)
    total_jd_keywords = sum(jd_counter.values())
    
    return min(100.0, (overlap_score / total_jd_keywords) * 100)


def calculate_skills_match(resume_skills: Set[str], jd_skills: Set[str]) -> float:
    """Calculate skills matching percentage"""
    if not jd_skills or not isinstance(jd_skills, set):
        return 100.0
    
    if not resume_skills or not isinstance(resume_skills, set):
        return 0.0
    
    matching_skills = resume_skills & jd_skills
    return (len(matching_skills) / len(jd_skills)) * 100


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    if not text or not isinstance(text, str):
        return []
    
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > MIN_SENTENCE_LENGTH]

# -------------------- CONTEXT & HELPERS --------------------
@contextmanager
def temporary_file(file_path: str):
    """Context manager for temporary file handling with proper cleanup"""
    try:
        yield file_path
    finally:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass  # File already deleted, that's fine
        except PermissionError as e:
            logger.warning(f"Permission denied deleting {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {file_path}: {e}")


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    if not filename or not isinstance(filename, str):
        return False
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF with improved performance and error handling"""
    try:
        text_parts = []
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = min(len(reader.pages), MAX_PDF_PAGES)
            
            for i in range(num_pages):
                try:
                    content = reader.pages[i].extract_text()
                    if content:
                        text_parts.append(content)
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {i+1}: {e}")
                    continue
        
        text = "\n".join(text_parts)
        extracted = text.strip()[:MAX_TEXT_LENGTH]
        
        if len(extracted) < MIN_TEXT_LENGTH:
            raise ValueError("Insufficient text extracted from PDF. Please ensure the PDF contains readable text.")
        
        return extracted
        
    except PyPDF2.errors.PdfReadError as e:
        raise ValueError(f"Invalid or corrupted PDF file: {str(e)}")
    except FileNotFoundError:
        raise ValueError("PDF file not found")
    except PermissionError:
        raise ValueError("Permission denied reading PDF file")
    except Exception as e:
        raise ValueError(f"Failed to process PDF: {str(e)}")


# -------------------- FLASK ROUTES --------------------
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_resume():
    """Analyze resume and predict job role"""
    try:
        if not model_manager.is_loaded():
            return jsonify({'error': 'Models not loaded. Please restart the server.'}), 500
        
        # Check if file is present
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file provided'}), 400
        
        file = request.files['resume']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PDF and TXT files are allowed.'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure upload folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        # Extract text
        with temporary_file(filepath):
            if filename.endswith('.pdf'):
                resume_text = extract_text_from_pdf(filepath)
            else:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    resume_text = f.read()[:MAX_TEXT_LENGTH]
            
            if len(resume_text) < MIN_TEXT_LENGTH:
                return jsonify({'error': 'Resume text too short. Please provide more content.'}), 400
            
            # Predict job role
            predicted_role = model_manager.resume_classifier.predict([resume_text])[0]
            confidence = max(model_manager.resume_classifier.predict_proba([resume_text])[0])
            
            # Predict salary
            salary_prediction = model_manager.salary_model.predict([resume_text])[0]
            
            # Extract skills
            skills = list(extract_skills(resume_text))
            
            return jsonify({
                'predicted_role': predicted_role,
                'confidence': float(confidence),
                'predicted_salary': float(salary_prediction),
                'extracted_skills': skills,
                'resume_length': len(resume_text)
            })
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error during analysis'}), 500


@app.route('/api/match', methods=['POST'])
def match_job_description():
    """Match resume against job descriptions"""
    try:
        if not model_manager.is_loaded():
            return jsonify({'error': 'Models not loaded'}), 500
        
        data = request.get_json()
        
        if not data or 'resume_text' not in data:
            return jsonify({'error': 'Resume text is required'}), 400
        
        resume_text = data['resume_text']
        
        if len(resume_text) < MIN_TEXT_LENGTH:
            return jsonify({'error': 'Resume text too short'}), 400
        
        # Encode resume
        resume_embedding = model_manager.embed_model.encode(resume_text, convert_to_tensor=True)
        
        # Calculate similarity
        similarities = util.cos_sim(resume_embedding, model_manager.job_embeddings)[0]
        
        # Get top matches
        top_indices = similarities.argsort(descending=True)[:TOP_MATCHES]
        
        matches = []
        for idx in top_indices:
            idx = idx.item()
            matches.append({
                'job_title': model_manager.job_df.iloc[idx]['Job Title'],
                'similarity_score': float(similarities[idx]),
                'job_description': model_manager.job_df.iloc[idx]['Job Description'][:200] + '...'
            })
        
        return jsonify({'matches': matches})
    
    except Exception as e:
        logger.error(f"Matching failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error during matching'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': model_manager.is_loaded()
    })


# -------------------- ERROR HANDLERS --------------------
@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


# -------------------- MAIN ENTRY --------------------
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    try:
        mode = os.getenv("MODEL_MODE", "fast").lower()
        quick_mode = mode == "fast"
        logger.info(f"Initializing AI Resume Analyzer in {'FAST' if quick_mode else 'ACCURATE'} mode...")
        model_manager.load_models(quick_mode)
        logger.info("System ready!")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        logger.error(traceback.format_exc())
        exit(1)
    app.run(debug=True, host='0.0.0.0', port=5000)'''