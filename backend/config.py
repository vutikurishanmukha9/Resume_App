"""
Configuration settings for AI Resume Analyzer
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# App config
UPLOAD_FOLDER = str(BASE_DIR / "uploads")
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

# Model files
JOB_CLASSIFIER_PATH = MODELS_DIR / "job_classifier.pkl"
SALARY_PREDICTOR_PATH = MODELS_DIR / "salary_predictor.pkl"
RESUME_CLASSIFIER_PATH = MODELS_DIR / "resume_classifier.pkl"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
EMBEDDING_CACHE_FILE = MODELS_DIR / "job_embeddings_cache.pkl"
EMBEDDING_HASH_FILE = MODELS_DIR / "job_embeddings_hash.txt"

# Data files
JOB_DATA_CSV = DATA_DIR / "job_title_des.csv"
SKILLS_TAXONOMY_PATH = DATA_DIR / "skills_taxonomy.json"

# Constants
MAX_TEXT_LENGTH = 5000
MAX_PDF_PAGES = 5
TOP_MATCHES = 3
MIN_TEXT_LENGTH = 50

# Matching weights
MATCHING_WEIGHTS = {
    'semantic': 0.40,
    'keyword': 0.30,
    'skills': 0.20,
    'context': 0.10
}

# Analytics
ANALYTICS_FILE = DATA_DIR / "analytics.json"
