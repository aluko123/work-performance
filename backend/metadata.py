"""
Lightweight metadata cache for the data corpus.
Computed once, cached in Redis, updated on new uploads.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any
import redis

from .database import SessionLocal
from . import db_models

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
METADATA_KEY = "corpus:metadata"
METADATA_TTL = 3600  # 1 hour cache


def get_corpus_metadata(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Get lightweight metadata about available data.
    Cached in Redis, refreshed hourly or on demand.
    
    Returns:
        {
            "date_range": {"min": "2024-06-10", "max": "2024-09-30"},
            "speakers": ["Tasha", "Mike", "Jordan", ...],
            "total_utterances": 2088,
            "metrics_available": ["SAFETY_Score", "QUALITY_Score", ...]
        }
    """
    try:
        r = redis.from_url(REDIS_URL)
        
        # Try cache first
        if not force_refresh:
            cached = r.get(METADATA_KEY)
            if cached:
                return json.loads(cached)
        
        # Compute fresh metadata
        session = SessionLocal()
        try:
            from sqlalchemy import func
            
            # Get date range (indexed query - fast even with millions of rows)
            date_stats = session.query(
                func.min(db_models.Utterance.date).label('min_date'),
                func.max(db_models.Utterance.date).label('max_date'),
                func.count(db_models.Utterance.id).label('count')
            ).first()
            
            # Get unique speakers (limited to 20 most frequent)
            speakers = session.query(db_models.Utterance.speaker)\
                .filter(db_models.Utterance.speaker.isnot(None))\
                .distinct()\
                .limit(20)\
                .all()
            
            # Build metadata
            metadata = {
                "date_range": {
                    "min": str(date_stats.min_date) if date_stats.min_date else None,
                    "max": str(date_stats.max_date) if date_stats.max_date else None
                },
                "speakers": [s[0] for s in speakers if s[0]],
                "total_utterances": date_stats.count or 0,
                "last_updated": datetime.utcnow().isoformat()
            }
            
            # Cache it
            r.setex(METADATA_KEY, METADATA_TTL, json.dumps(metadata))
            
            return metadata
            
        finally:
            session.close()
    
    except Exception as e:
        print(f"Failed to get metadata: {e}")
        # Return safe defaults
        return {
            "date_range": {"min": None, "max": None},
            "speakers": [],
            "total_utterances": 0
        }


def invalidate_metadata_cache():
    """Call this after uploading new data"""
    try:
        r = redis.from_url(REDIS_URL)
        r.delete(METADATA_KEY)
        print("📊 Metadata cache invalidated")
    except Exception as e:
        print(f"Failed to invalidate cache: {e}")
