"""
Custom Exceptions for AI Resume Analyzer

Provides a hierarchy of application-specific exceptions for better error handling.
"""


class ResumeAnalyzerError(Exception):
    """Base exception for all Resume Analyzer errors."""
    
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)
    
    def to_dict(self) -> dict:
        """Convert exception to JSON-serializable dict."""
        return {
            'error': self.__class__.__name__,
            'message': self.message,
            'status_code': self.status_code
        }


class ModelNotLoadedError(ResumeAnalyzerError):
    """Raised when ML models are not yet loaded."""
    
    def __init__(self, message: str = "Models are still loading. Please try again in a moment."):
        super().__init__(message, status_code=503)


class InvalidFileError(ResumeAnalyzerError):
    """Raised when uploaded file is invalid."""
    
    def __init__(self, message: str = "Invalid file format or content."):
        super().__init__(message, status_code=400)


class FileTooLargeError(ResumeAnalyzerError):
    """Raised when uploaded file exceeds size limit."""
    
    def __init__(self, message: str = "File exceeds maximum size limit."):
        super().__init__(message, status_code=413)


class TextExtractionError(ResumeAnalyzerError):
    """Raised when text cannot be extracted from file."""
    
    def __init__(self, message: str = "Failed to extract text from file."):
        super().__init__(message, status_code=422)


class InsufficientTextError(ResumeAnalyzerError):
    """Raised when extracted text is too short for analysis."""
    
    def __init__(self, message: str = "Resume text is too short for meaningful analysis."):
        super().__init__(message, status_code=422)


class AnalysisError(ResumeAnalyzerError):
    """Raised when resume analysis fails."""
    
    def __init__(self, message: str = "Failed to analyze resume."):
        super().__init__(message, status_code=500)


class RateLimitExceededError(ResumeAnalyzerError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded. Please try again later."):
        super().__init__(message, status_code=429)
