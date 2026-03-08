"""
Analytics Service for AI Resume Analyzer

Tracks usage analytics for uploads, JD matches, and ATS scores.
Uses a threading lock to prevent concurrent writes from corrupting
the JSON file.
"""

import json
import logging
import os
import threading
from typing import Dict, Any
from datetime import datetime

from backend.config import ANALYTICS_FILE

logger = logging.getLogger(__name__)

_analytics_lock = threading.Lock()


def track_analysis(analysis_type: str, data: Dict[str, Any]) -> None:
    """Track analysis for analytics purposes (thread-safe)."""
    try:
        with _analytics_lock:
            # Load existing analytics
            if os.path.exists(ANALYTICS_FILE):
                with open(ANALYTICS_FILE, 'r') as f:
                    analytics = json.load(f)
            else:
                analytics = {
                    'analyses': [],
                    'summary': {
                        'total_uploads': 0,
                        'total_matches': 0,
                        'avg_match_score': 0.0,
                        'last_updated': None
                    }
                }

            # Add new analysis
            analysis_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': analysis_type,
                'data': data
            }
            analytics['analyses'].append(analysis_entry)

            # Update summary
            if analysis_type == 'upload':
                analytics['summary']['total_uploads'] += 1
            elif analysis_type in ('jd_match', 'analyze_full'):
                analytics['summary']['total_matches'] += 1
                # Update average match score
                match_scores = [a['data'].get('match_percentage', 0)
                              for a in analytics['analyses']
                              if a['type'] in ('jd_match', 'analyze_full')]
                if match_scores:
                    analytics['summary']['avg_match_score'] = sum(match_scores) / len(match_scores)

            analytics['summary']['last_updated'] = datetime.now().isoformat()

            # Save analytics (keep last 1000 entries)
            analytics['analyses'] = analytics['analyses'][-1000:]

            # Write atomically: write to temp then rename
            tmp_path = str(ANALYTICS_FILE) + '.tmp'
            with open(tmp_path, 'w') as f:
                json.dump(analytics, f, indent=2)
            os.replace(tmp_path, str(ANALYTICS_FILE))

    except Exception as e:
        logger.warning(f"Failed to track analytics: {e}")
