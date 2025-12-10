# AI Resume Analyzer

An AI-powered web application that analyzes resumes, matches them with job descriptions, calculates ATS scores, and predicts the best-fit job titles using advanced NLP and machine learning techniques.

---

## Features

### Core Capabilities

- **Resume Upload** - Supports PDF and TXT formats for easy resume submission
- **Job Description Analysis** - Input any job description to check compatibility with detailed breakdowns
- **AI-Powered Matching** - Uses state-of-the-art NLP models for:
  - Predicted job title that best fits your profile
  - Skill match percentage between your resume and the JD
- **ATS Score Analysis** - Comprehensive Applicant Tracking System scoring with:
  - Overall ATS compatibility score (0-100)
  - Sub-scores for keywords, skills, experience, and education
  - Quick Scan and Deep Analysis modes
  - Matched and missing keywords identification
  - Actionable improvement suggestions
- **Light/Dark Theme** - Toggle between dark and light modes with theme persistence
- **Modern UI/UX** - Features glassmorphism effects, smooth animations, custom scrollbars, and responsive design
- **Real-Time Analysis** - Receive instantaneous results upon submission without page reloads

### Advanced Features

- **Skills Extraction** - Detects 200+ technical skills across 13 categories with intelligent variation handling (e.g., "Node" to "Node.js", "K8s" to "Kubernetes")
- **Missing Keywords Detection** - Identifies keywords from job descriptions that are missing from your resume, ranked by importance:
  - Critical (High Priority)
  - Important (Medium Priority)
  - Optional (Low Priority)
- **Actionable Recommendations** - Provides specific suggestions on which keywords and skills to add
- **Skills Breakdown** - Categorized view of matched vs. missing skills by technology domain
- **Achievement Detection** - Identifies quantifiable achievements in your resume
- **Experience Analysis** - Extracts and analyzes years of experience
- **Education Detection** - Identifies education level (Bachelor's, Master's, PhD)
- **Analytics Tracking** - Logs usage patterns and model performance for continuous improvement
- **Rate Limiting** - Prevents abuse with request throttling protection

---

## Screenshots

### Dark Mode (Default)
The application features a sleek dark theme with gradient accents and glassmorphism effects.

### Light Mode
A clean, professional light theme with soft colors and pleasant aesthetics.

### ATS Score Analysis
Comprehensive ATS scoring with detailed breakdown of sub-scores, matched keywords, and improvement suggestions.

---

## Technology Stack

| Category | Technologies | Description |
|:---------|:-------------|:------------|
| Backend | Flask (Python) | Lightweight web framework for the REST API |
| Frontend | HTML5, CSS3, JavaScript | Responsive UI with glassmorphism and animations |
| AI/NLP Models | Sentence Transformers, Scikit-learn | Text embedding, similarity, and job title prediction |
| Libraries | PyPDF2, NumPy, Pandas, Joblib, Flask-Limiter, Flask-CORS | PDF parsing, data manipulation, model serialization |
| Deployment | Render, Railway, Docker | Cloud deployment with containerization support |

---

## Project Structure

```
Resume_App/
├── backend/                    # Python backend code
│   ├── app.py                  # Main Flask application
│   ├── config.py               # Configuration settings
│   ├── services/               # Business logic modules
│   │   ├── model_manager.py    # ML model loading and management
│   │   └── ats_scorer.py       # ATS scoring engine
│   ├── routes/                 # Route blueprints (future expansion)
│   └── utils/                  # Utility functions
│
├── frontend/                   # Frontend assets
│   ├── static/
│   │   ├── style.css           # Dark/Light theme styling
│   │   ├── script.js           # Frontend logic and API handling
│   │   └── favicon.png         # Application favicon
│   └── templates/
│       └── index.html          # Single-page application interface
│
├── models/                     # Machine learning models
│   ├── job_classifier.pkl      # Trained model for job prediction
│   ├── resume_classifier.pkl   # Resume classification model
│   ├── tfidf_vectorizer.pkl    # TF-IDF vectorizer
│   └── job_embeddings_cache.pkl # Precomputed embeddings cache
│
├── data/                       # Data files
│   ├── job_title_des.csv       # Job titles and descriptions dataset
│   ├── skills_taxonomy.json    # Skills database (200+ skills)
│   └── analytics.json          # Usage analytics data
│
├── uploads/                    # Temporary storage for uploaded resumes
├── run.py                      # Application entry point
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container build instructions
├── Procfile                    # Process file for cloud deployment
├── render.yaml                 # Render deployment configuration
├── railway.json                # Railway deployment configuration
└── README.md
```

---

## Local Development Setup

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

1. **Clone the Repository**

```bash
git clone https://github.com/vutikurishanmukha9/Resume_App.git
cd Resume_App
```

2. **Create Virtual Environment**

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Start the Application**

```bash
python run.py
```

5. **Access the Application**

Open your browser and navigate to: `http://127.0.0.1:5000/`

---

## How It Works

### Analysis Pipeline

1. **Input** - User uploads a resume (PDF or TXT) and optionally provides a job description
2. **Extraction** - Text is extracted using PyPDF2 (for PDFs) or standard text reading
3. **Feature Extraction** - The system extracts key features from the resume:
   - Years of experience (from text patterns and date ranges)
   - Education level (Bachelor's, Master's, PhD)
   - Seniority level (Entry, Mid, Senior, Lead)
   - Technical skills (200+ skills across 13 categories)
   - Quantifiable achievements
4. **Embedding** - The Sentence Transformer model converts text into numerical vector embeddings
5. **Matching** - Cosine similarity is calculated to determine the skill match percentage
6. **Prediction** - A trained ML model predicts the best-fit job title
7. **ATS Scoring** - Comprehensive ATS compatibility analysis with weighted scoring
8. **Insights** - The system provides detailed breakdowns, missing keywords, and suggestions

### ATS Scoring System

The ATS Scorer analyzes resumes against job descriptions using multiple factors:

| Component | Weight | Description |
|:----------|:-------|:------------|
| Keywords Match | 35% | Presence of required and preferred keywords |
| Skills Match | 25% | Technical skills alignment with requirements |
| Experience Match | 20% | Years of experience vs. requirements |
| Education Match | 10% | Education level alignment |
| Formatting | 10% | Resume structure and section detection |

**Analysis Modes:**
- **Quick Scan** - Fast keyword-based analysis for rapid feedback
- **Deep Analysis** - Comprehensive analysis including semantic matching, achievement detection, and detailed recommendations

### Skills Taxonomy

The application uses a comprehensive skills taxonomy with 200+ technical skills organized into 13 categories:

- Programming Languages (Python, Java, JavaScript, C++, etc.)
- Web Frameworks (React, Angular, Node.js, Django, etc.)
- Databases (MySQL, PostgreSQL, MongoDB, Redis, etc.)
- Cloud Platforms (AWS, Azure, GCP)
- DevOps Tools (Docker, Kubernetes, Jenkins, Terraform, etc.)
- Data Science and ML (TensorFlow, PyTorch, Scikit-learn, etc.)
- Mobile Development (React Native, Flutter, Swift, Kotlin)
- Testing Frameworks (Jest, Pytest, Selenium, etc.)
- Methodologies (Agile, Scrum, DevOps, CI/CD)

Each skill supports multiple variations for robust detection.

---

## Deployment

### Render

The application includes a `render.yaml` blueprint for one-click deployment on Render.com.

### Railway

Configuration files `railway.json` and `nixpacks.toml` are included for Railway deployment.

### Docker

Build and run using Docker:

```bash
docker build -t resume-analyzer .
docker run -p 5000:5000 resume-analyzer
```

---

## API Endpoints

| Endpoint | Method | Description |
|:---------|:-------|:------------|
| `/` | GET | Main application interface |
| `/upload` | POST | Upload and analyze resume |
| `/match_jd_resume` | POST | Match resume with job description |
| `/ats_score` | POST | Calculate ATS score for resume against JD |
| `/health` | GET | Health check endpoint |
| `/ready` | GET | Model readiness check |

---

## Rate Limiting

To ensure fair usage, the application implements rate limiting:

- 10 requests per minute per endpoint
- 50 requests per hour (default)
- 200 requests per day (default)

When limits are exceeded, users receive a friendly error message with retry information.

---

## Theme Customization

The application supports two themes:

- **Dark Mode** (Default) - Sleek dark theme with gradient accents, custom scrollbars, and glassmorphism effects
- **Light Mode** - Clean, professional light theme with soft colors and pleasant aesthetics

Theme preference is automatically saved and persists across browser sessions.

---

## Author

**Vutikuri Shanmukha**

B.Tech in Electronics and Communication Engineering

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgements

- Sentence Transformers Team for the NLP models
- Flask Community for the web framework
- Scikit-learn Contributors for the ML library
