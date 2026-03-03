"""
Analytics Service for AI Resume Analyzer

Tracks usage analytics for uploads, JD matches, and ATS scores.
"""

import os
import json
import logging
from typing import Dict, Any
from datetime import datetime

from backend.config import ANALYTICS_FILE

logger = logging.getLogger(__name__)


def track_analysis(analysis_type: str, data: Dict[str, Any]) -> None:
    """Track analysis for analytics purposes"""
    try:
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
        elif analysis_type == 'jd_match':
            analytics['summary']['total_matches'] += 1
            # Update average match score
            match_scores = [a['data'].get('match_percentage', 0) 
                          for a in analytics['analyses'] 
                          if a['type'] == 'jd_match']
            if match_scores:
                analytics['summary']['avg_match_score'] = sum(match_scores) / len(match_scores)
        
        analytics['summary']['last_updated'] = datetime.now().isoformat()
        
        # Save analytics (keep last 1000 entries)
        analytics['analyses'] = analytics['analyses'][-1000:]
        
        with open(ANALYTICS_FILE, 'w') as f:
            json.dump(analytics, f, indent=2)
            
    except Exception as e:
        logger.warning(f"Failed to track analytics: {e}")
