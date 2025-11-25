# AI Resume Analyzer

An AI-powered web application that **analyzes resumes**, **matches them with job descriptions**, and **predicts the best-fit job titles and estimated salaries** all in one intuitive interface.

-----

## 🚀 Features

This application leverages advanced NLP to provide comprehensive insights into job fit and career potential.

  * ✅ **Resume Upload** — Supports **PDF and TXT** formats for easy resume submission.
  * ✅ **Job Description Analysis** — Input any job description (JD) to check compatibility.
  * ✅ **AI-Powered Matching** — Uses state-of-the-art NLP models for core predictions:
      * **Predicted Job Title** that best fits your profile.
      * **Skill Match Percentage** between your resume and the JD.
      * **Estimated Salary Range** for the predicted role.
  * ✅ **Smart Suggestions** — Highlights **missing keywords** and provides actionable improvements to optimize your resume for **ATS (Applicant Tracking Systems)**.
  * ✅ **Modern UI/UX** — Features a stunning **Dark Theme** with **Glassmorphism** effects, smooth animations, and a responsive design.
  * ✅ **Real-Time Insights** — Receive instantaneous analysis and results upon submission without page reloads.

-----

## 🧠 Tech Stack

| Category | Technologies | Description |
| :--- | :--- | :--- |
| **Backend** | **Flask (Python)** | Robust and lightweight web framework for the backend. |
| **Frontend** | HTML5, CSS3, JavaScript | Modern, responsive UI with Glassmorphism and animations. |
| **AI/NLP Models** | **Sentence Transformers, Scikit-learn** | Core components for text embedding, similarity, and job title prediction. |
| **Libraries Used** | PyPDF2, NumPy, Pandas, Joblib | PDF parsing, data manipulation, and model serialization. |
| **Deployment** | Render / Railway / AWS | Recommended platforms for production deployment. |

-----

## ⚙️ Local Setup

Follow these steps to get the **AI Resume Analyzer** running on your local machine.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/ai-resume-analyzer.git
cd ai-resume-analyzer
```

### 2️⃣ Create and Activate Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

**For Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**For Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

All required Python packages are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the App

Start the Flask application server.

```bash
python app.py
```

### 5️⃣ Open in Browser

The application will be accessible locally.

```bash
Visit 👉 http://127.0.0.1:5000/
```

-----

## 📁 Project Structure

```
AI-Resume-Analyzer/
│
├── app.py                      # Main Flask backend file
├── config.py                   # Configuration settings
├── job_classifier.pkl          # Trained model for job prediction
├── salary_predictor.pkl        # Trained model for salary estimation
├── job_title_des.csv           # Dataset for job titles and descriptions
│
├── static/
│   ├── style.css               # Modern Dark Theme Styling
│   └── script.js               # Frontend logic and API handling
│
├── templates/
│   └── index.html              # Single-page application interface
│
├── uploads/                    # Temporary storage for uploaded resumes
├── requirements.txt
└── README.md
```

-----

## 🧩 How It Works

1.  **Input:** User uploads a **resume (PDF or TXT)** and optionally provides a **job description (JD)**.
2.  **Extraction:** Text is extracted from the file using **PyPDF2** (for PDFs) or standard text reading.
3.  **Embedding:** The **SentenceTransformer** model converts the resume and JD text into numerical vector **embeddings**.
4.  **Matching:** **Cosine similarity** is calculated between the embeddings to determine the **Skill Match Percentage**.
5.  **Prediction:** A trained **Machine Learning model** predicts the best-fit **Job Title**.
6.  **Insights:** The system provides **salary estimation** and **smart suggestions** for resume improvement.

### 🧪 Example Output

| Metric | Result |
| :--- | :--- |
| **Predicted Job Title** | Data Scientist |
| **Match Score** | **86%** |
| **Estimated Salary** | ₹8.2 – ₹9.5 LPA |
| **Suggestions** | Add keywords: "Deep Learning", "Model Optimization" |

-----

## 💡 Future Enhancements

We are always looking to improve! Potential future features include:

  * 🗣️ Voice Resume Input for accessibility.
  * 🧬 Integration with large language models (OpenAI / Gemini) for detailed resume rewrites.
  * 🌐 Multi-language resume parsing capabilities.
  * 📈 Data visualization of career insights and trends.
  * 🧾 Auto-suggest and implement resume corrections.

-----

## 👨‍💻 Author

Developed by: **Vutikuri Shanmukha**

  * 📍 B.Tech in Electronics & Communication Engineering
  * 💼 Passionate about AI, NLP, and Human–Machine synergy

-----

## 🪶 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute it for any purpose.

-----

## 🌟 Acknowledgements

A special thanks to the following for their contributions and resources:

  * **Sentence Transformers Team** for the powerful NLP models.
  * **Flask Community** for the robust and versatile web framework.
  * **OpenAI** for inspiration on AI-based resume screening solutions.
