# AI Resume Analyzer

An AI-powered web application that analyzes resumes, matches them with job descriptions, calculates ATS scores, and predicts best-fit job titles using NLP and machine learning.

---

## Features

### Core Capabilities

- **Resume Upload** — PDF and TXT formats, parsed in-memory (no disk writes)
- **Job Description Analysis** — Input any JD to check compatibility with detailed breakdowns
- **Unified Analysis** — Single `/analyze-full` endpoint: one upload, one parse, all results
- **AI-Powered Matching** — Sentence Transformers for:
  - Predicted job title
  - Semantic similarity scoring
  - Contextual sentence-level matching (skips contact info / headers)
- **ATS Score Analysis** — Comprehensive scoring with:
  - Overall ATS compatibility score (0–100)
  - Sub-scores for keywords, skills, experience, education, and formatting
  - Quick Scan and Deep Analysis modes
  - Negation-aware keyword classification ("NOT required" handled correctly)
  - Matched/missing keywords identification
  - Actionable improvement suggestions
- **Salary Prediction** — Role-adjusted salary estimates based on experience, education, seniority, and skills
- **Light/Dark Theme** — Toggle with persistence
- **Modern UI/UX** — Glassmorphism, smooth animations, responsive design

### Advanced Features

- **Skills Extraction** — 200+ technical skills across 13 categories with variation handling (e.g., "K8s" → "Kubernetes")
- **Missing Keywords Detection** — Ranked by importance (Critical / Important / Optional)
- **Achievement Detection** — Identifies quantifiable achievements with proximity-based verb-metric matching
- **Experience Analysis** — Section-aware month-level date parsing
- **Education Detection** — False-positive-resistant (handles "Scrum Master", "Mastered Python" correctly)
- **Expanded Job Ontology** — 22 categories across Tech, Healthcare, Finance, Engineering, Design, HR, and more
- **Rate Limiting** — Request throttling protection
- **Analytics Tracking** — Usage pattern logging

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  React Frontend (Vite + TypeScript + Tailwind)          │
│  Single API call to /analyze-full                       │
└──────────────────────┬──────────────────────────────────┘
                       │ POST (file + JD)
                       ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI Backend                                        │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ read_upload   │  │ extract_text │  │ run_in_      │  │
│  │ _bytes()     │→ │ _from_bytes()│→ │ threadpool() │  │
│  │ (chunked,    │  │ (BytesIO,    │  │ (non-blocking│  │
│  │  16MB limit) │  │  pdfplumber) │  │  CPU work)   │  │
│  └──────────────┘  └──────────────┘  └──────┬───────┘  │
│                                              │          │
│                    ┌─────────────────────────┐│          │
│                    │ _run_full_analysis()    ││          │
│                    │ ├─ analyze_resume()     ││          │
│                    │ ├─ calculate_jd_match() ││          │
│                    │ └─ ATSScorer.score()    ││          │
│                    └─────────────────────────┘│          │
└─────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|:---------|:----------|
| **In-memory parsing (BytesIO)** | No temp files → cloud-ready (Lambda, Heroku, Render) |
| **Unified endpoint** | 1 upload instead of 3 → 66% less bandwidth/CPU |
| **run_in_threadpool** | Non-blocking async for CPU-bound ML inference |
| **Synchronous model loading** | No 503 race condition — server is ready before accepting traffic |
| **Role-adjusted salary** | Job category multiplier applied after base prediction |
| **Substantive sentence filter** | Contextual similarity skips contact info and "About Us" headers |
| **Custom exception hierarchy** | Clean error propagation instead of string-matching error messages |

---

## Technology Stack

| Category | Technologies |
|:---------|:-------------|
| Backend | FastAPI, Python 3.10, Uvicorn |
| Frontend | React, Vite, TypeScript, Tailwind CSS, Lucide React |
| AI/NLP | Sentence Transformers (all-MiniLM-L6-v2), Scikit-learn |
| PDF Parsing | pdfplumber (in-memory via BytesIO) |
| Libraries | NumPy, Pandas, Joblib, SlowAPI |
| Deployment | Docker, Render, Railway, Nixpacks |

---

## Project Structure

```
Resume_App/
├── backend/
│   ├── main.py                 # FastAPI app, lifespan, error handlers
│   ├── config.py               # Configuration settings
│   ├── exceptions.py           # Custom exception hierarchy
│   ├── rate_limiter.py         # Rate limiting config
│   ├── services/
│   │   ├── model_manager.py    # ML model loading (synchronous)
│   │   ├── analysis.py         # Resume analysis, JD matching, salary prediction
│   │   ├── ats_scorer.py       # ATS scoring engine (22 job categories)
│   │   └── analytics.py        # Usage analytics
│   ├── routes/
│   │   ├── analyze.py          # /analyze-full — unified endpoint
│   │   ├── upload.py           # /upload — resume analysis
│   │   ├── ats.py              # /ats_score — ATS scoring
│   │   ├── match.py            # /match_jd_resume — JD matching
│   │   └── general.py          # /health, /ready
│   └── utils/
│       ├── text_processing.py  # BytesIO extraction, chunked upload
│       ├── feature_extractor.py# Experience, education, seniority extraction
│       ├── keyword_extractor.py# Keyword extraction and overlap
│       └── skill_extractor.py  # 200+ skills across 13 categories
│
├── frontend/
│   ├── src/
│   │   ├── components/         # React UI components
│   │   ├── hooks/              # Custom React hooks
│   │   └── lib/
│   │       ├── api.ts          # Single /analyze-full API call
│   │       └── types.ts        # TypeScript interfaces
│   ├── package.json
│   └── vite.config.ts
│
├── models/                     # ML models (.pkl files)
├── data/                       # Job dataset, skills taxonomy, analytics
├── Dockerfile                  # Multi-stage build (Node + Python)
├── render.yaml                 # Render deployment config
├── railway.json                # Railway deployment config
├── nixpacks.toml               # Nixpacks config (Railway fallback)
├── Procfile                    # Process file for cloud platforms
├── requirements.txt            # Python dependencies
└── run.py                      # Development entry point
```

---

## Local Development

### Prerequisites

- Python 3.10+
- Node.js 18+
- pip

### Setup

1. **Clone and set up backend**

```bash
git clone https://github.com/vutikurishanmukha9/Resume_App.git
cd Resume_App
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
python run.py
```

2. **Start the frontend** (new terminal)

```bash
cd frontend
npm install
npm run dev
```

3. **Open the app** at `http://localhost:5173/` (backend at `http://localhost:5000/`)

---

## API Endpoints

| Endpoint | Method | Description |
|:---------|:-------|:------------|
| `/analyze-full` | POST | **Unified** — upload once, get all results (recommended) |
| `/upload` | POST | Resume analysis only (job prediction, salary) |
| `/match_jd_resume` | POST | JD-resume match with skills breakdown |
| `/ats_score` | POST | ATS score with keyword/achievement analysis |
| `/health` | GET | Health check |
| `/ready` | GET | Model readiness check |

### `/analyze-full` Request

```
POST /analyze-full
Content-Type: multipart/form-data

resume: <file>
jd_text: <string>
jd_title: <string> (optional)
mode: "quick" | "deep" (default: "deep")
```

### Response

```json
{
  "success": true,
  "upload": { "predicted_job": "...", "salary": "...", ... },
  "ats": { "ats_score": 72, "sub_scores": {...}, ... },
  "jd_match": { "match_percentage": 68.5, "component_scores": {...}, ... }
}
```

---

## ATS Scoring System

| Component | Weight | Description |
|:----------|:-------|:------------|
| Keywords Match | 35% | Required/preferred keywords with negation detection |
| Skills Match | 25% | Technical skills alignment across 13 categories |
| Experience Match | 20% | Years of experience vs. requirements |
| Education Match | 10% | Education level with false-positive resistance |
| Formatting | 10% | Resume structure and section detection |

**Modes:** Quick Scan (keyword-based) · Deep Analysis (semantic + achievements + detailed recommendations)

---

## Deployment

### Docker

```bash
docker build -t resume-analyzer .
docker run -p 7860:7860 resume-analyzer
```

### Render

Includes `render.yaml` blueprint for one-click deployment.

### Railway

Includes `railway.json` and `nixpacks.toml` for Railway deployment.

### Production Notes

- Uses single worker (`--workers 1`) to avoid duplicating ML models in memory
- Models load synchronously at startup — container is not ready until models are loaded
- Use `/ready` endpoint for container health checks (not `/health`)
- No disk writes — fully stateless, works on ephemeral filesystems

---

## Author

**Vutikuri Shanmukha**
B.Tech in Electronics and Communication Engineering

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Sentence Transformers](https://www.sbert.net/) for NLP models
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [pdfplumber](https://github.com/jsvine/pdfplumber) for PDF text extraction
- [Scikit-learn](https://scikit-learn.org/) for ML models
