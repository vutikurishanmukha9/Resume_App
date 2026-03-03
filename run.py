"""
AI Resume Analyzer - Application Entry Point
Run this file to start the FastAPI application.
"""
import os
import sys

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from backend.main import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting AI Resume Analyzer on port {port}...")
    uvicorn.run(app, host='0.0.0.0', port=port)
