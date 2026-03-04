"""
Text Processing Utilities for AI Resume Analyzer

Handles file validation, PDF/TXT text extraction, and temporary file management.
Supports both file-path and in-memory (BytesIO) extraction.
"""

import io
import os
import logging
import pdfplumber
from contextlib import contextmanager

from backend.config import MAX_TEXT_LENGTH, MAX_PDF_PAGES, MIN_TEXT_LENGTH, ALLOWED_EXTENSIONS

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 16 * 1024 * 1024  # 16 MB hard limit


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


async def read_upload_bytes(upload_file) -> bytes:
    """
    Read an uploaded file into memory in chunks with a hard size limit.

    Returns the raw bytes.  Raises ValueError if the file exceeds
    MAX_UPLOAD_BYTES.
    """
    chunks: list[bytes] = []
    total = 0
    chunk_size = 8 * 1024  # 8 KB

    while True:
        chunk = await upload_file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise ValueError(
                f'File too large ({total // (1024*1024)}+ MB). '
                f'Maximum allowed size is {MAX_UPLOAD_BYTES // (1024*1024)} MB.'
            )
        chunks.append(chunk)

    return b''.join(chunks)


# ─── PDF extraction ───────────────────────────────────────────────

def _extract_pdf_text(source) -> str:
    """
    Extract text from a PDF.

    *source* can be a file path (str) **or** a BytesIO / file-like object.
    pdfplumber.open() accepts both transparently.
    """
    try:
        text = ""
        with pdfplumber.open(source) as pdf:
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
            raise ValueError(
                "Insufficient text extracted from PDF. "
                "Please ensure the PDF contains readable text."
            )
        return extracted

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to process PDF: {str(e)}")


# ─── Public API ───────────────────────────────────────────────────

def extract_text_from_bytes(data: bytes, filename: str) -> str:
    """
    Extract text from raw file bytes (no disk I/O).

    Accepts the file content as *bytes* and the original *filename*
    (used only to determine the extension).
    """
    ext = filename.rsplit('.', 1)[-1].lower()

    if ext == 'pdf':
        return _extract_pdf_text(io.BytesIO(data))
    elif ext == 'txt':
        content = data.decode('utf-8', errors='ignore').strip()[:MAX_TEXT_LENGTH]
        if len(content) < MIN_TEXT_LENGTH:
            raise ValueError(
                "Text file is too short. "
                "Please provide a resume with at least 50 characters."
            )
        return content
    else:
        raise ValueError("Unsupported file type")


def extract_text_from_pdf(file_path: str) -> str:
    """Legacy wrapper — extracts text from a PDF file on disk."""
    return _extract_pdf_text(file_path)


def extract_text_from_file(file_path: str, filename: str) -> str:
    """Legacy wrapper — extracts text from a file on disk."""
    ext = filename.rsplit('.', 1)[-1].lower()

    if ext == 'pdf':
        return extract_text_from_pdf(file_path)
    elif ext == 'txt':
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()[:MAX_TEXT_LENGTH]
        except UnicodeDecodeError:
            raise ValueError(
                "Failed to read text file. "
                "Please ensure it's a valid UTF-8 encoded text file."
            )

        if len(content) < MIN_TEXT_LENGTH:
            raise ValueError(
                "Text file is too short. "
                "Please provide a resume with at least 50 characters."
            )
        return content
    else:
        raise ValueError("Unsupported file type")
