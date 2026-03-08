"""
Model Manager - Centralized ML model loading and management
"""
import os
import logging
import pickle
import traceback
import threading

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer

from backend.config import (
    JOB_CLASSIFIER_PATH,
    SALARY_PREDICTOR_PATH,
    JOB_DATA_CSV,
    EMBEDDING_CACHE_FILE
)

logger = logging.getLogger(__name__)


class ModelManager:
    """Centralized model management with caching and error handling"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
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
            if not os.path.exists(JOB_CLASSIFIER_PATH):
                raise FileNotFoundError(f"job_classifier.pkl not found at {JOB_CLASSIFIER_PATH}")
            self.resume_classifier = joblib.load(JOB_CLASSIFIER_PATH)
            logger.info("Resume classifier loaded")

            # Load salary predictor
            if not os.path.exists(SALARY_PREDICTOR_PATH):
                raise FileNotFoundError(f"salary_predictor.pkl not found at {SALARY_PREDICTOR_PATH}")
            self.salary_model = joblib.load(SALARY_PREDICTOR_PATH)
            logger.info("Salary predictor loaded")

            # Load job dataset
            if not os.path.exists(JOB_DATA_CSV):
                raise FileNotFoundError(f"job_title_des.csv not found at {JOB_DATA_CSV}")
            self.job_df = pd.read_csv(JOB_DATA_CSV)

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
            self._precompute_job_embeddings()

            self._models_loaded = True
            logger.info("All models successfully initialized")

        except FileNotFoundError as e:
            logger.error(f"Required file missing: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to initialize models: {str(e)}")

    def _precompute_job_embeddings(self):
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


# Singleton instance
model_manager = ModelManager()


def load_all_models():
    """Load all ML models synchronously during startup.
    
    Raises on failure so FastAPI's lifespan handler aborts
    instead of starting a silently broken server.
    """
    try:
        logger.info("Starting model loading...")
        model_manager.load_models()
        logger.info("All models loaded successfully!")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        logger.error(traceback.format_exc())
        raise  # Let lifespan handler abort startup


# Backward-compat alias
load_models_background = load_all_models
