"""
Text Processing Utilities for AI Resume Analyzer

Handles file validation, PDF/TXT text extraction, and temporary file management.
"""

import os
import logging
import pdfplumber
from contextlib import contextmanager

from backend.config import MAX_TEXT_LENGTH, MAX_PDF_PAGES, MIN_TEXT_LENGTH, ALLOWED_EXTENSIONS

logger = logging.getLogger(__name__)


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
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


MAX_UPLOAD_BYTES = 16 * 1024 * 1024  # 16 MB hard limit


async def save_upload_safely(upload_file, dest_path: str) -> None:
    """
    Save an uploaded file to disk in chunks with a hard size limit.
    
    Reads in 8 KB chunks and aborts immediately if the file exceeds
    MAX_UPLOAD_BYTES.  This prevents OOM when many users upload
    large PDFs concurrently.
    
    Raises:
        ValueError: if the file exceeds the size limit.
    """
    chunk_size = 8 * 1024  # 8 KB
    bytes_written = 0
    with open(dest_path, 'wb') as f:
        while True:
            chunk = await upload_file.read(chunk_size)
            if not chunk:
                break
            bytes_written += len(chunk)
            if bytes_written > MAX_UPLOAD_BYTES:
                # Clean up the partial file
                f.close()
                try:
                    os.remove(dest_path)
                except OSError:
                    pass
                raise ValueError(
                    f'File too large ({bytes_written // (1024*1024)}+ MB). '
                    f'Maximum allowed size is {MAX_UPLOAD_BYTES // (1024*1024)} MB.'
                )
            f.write(chunk)


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from PDF file using pdfplumber.
    Handles multi-column layouts and complex formatting better than PyPDF2.
    """
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            num_pages = min(len(pdf.pages), MAX_PDF_PAGES)

            for i in range(num_pages):
                try:
                    page = pdf.pages[i]
                    content = page.extract_text(
                        x_tolerance=3,
                        y_tolerance=3,
                    )
                    if content:
                        text += content + "\n"
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {i+1}: {e}")
                    continue

        extracted = text.strip()[:MAX_TEXT_LENGTH]

        if len(extracted) < MIN_TEXT_LENGTH:
            raise ValueError("Insufficient text extracted from PDF. Please ensure the PDF contains readable text.")

        return extracted

    except Exception as e:
        if "Insufficient text" in str(e):
            raise
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
