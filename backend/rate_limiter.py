"""
Rate Limiter Configuration for AI Resume Analyzer

Provides a shared slowapi Limiter instance used by main.py and route files.
"""

import logging

logger = logging.getLogger(__name__)

try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address

    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"]
    )
    rate_limiting_enabled = True
    logger.info("Rate limiting configured")
except ImportError:
    limiter = None
    rate_limiting_enabled = False
    logger.warning("slowapi not installed. Rate limiting disabled.")
