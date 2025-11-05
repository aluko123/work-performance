"""
Dynamic metric discovery and display-name mapping.

Sources (in priority order):
- METRICS_JSON_PATH env var (e.g., data/column_name_mapping.json)
- frontend-v2/public/column_name_mapping.json
- backend/config/chart_config.py (fallback)
"""

import json
import os
from functools import lru_cache
from typing import Dict, List, Tuple

from .config.chart_config import METRIC_DISPLAY_NAMES as FALLBACK_DISPLAY
from .config.chart_config import AGGREGATED_METRICS


@lru_cache(maxsize=1)
def _load_mapping() -> Dict[str, str]:
    """Load metric display mapping from JSON if available.

    The JSON format is expected to be: { "metric_key": { "original_name": "Nice Name", ...}, ... }
    Returns a dict of {metric_key: display_name}.
    """
    candidates = []
    env_path = os.getenv("METRICS_JSON_PATH")
    if env_path:
        candidates.append(env_path)
    # Common repo paths
    candidates.extend([
        "data/column_name_mapping.json",
        "frontend-v2/public/column_name_mapping.json",
        "frontend/column_name_mapping.json",
        "bert_classification/column_name_mapping.json",
    ])

    for path in candidates:
        try:
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            mapping: Dict[str, str] = {}
            # normalize keys and extract display name
            for k, v in raw.items():
                name = None
                if isinstance(v, dict):
                    name = v.get("original_name") or v.get("name")
                elif isinstance(v, str):
                    name = v
                if name:
                    mapping[k] = name
            if mapping:
                # Merge in any fallback display names not present (e.g., aggregated metrics)
                for k, v in FALLBACK_DISPLAY.items():
                    mapping.setdefault(k, v)
                return mapping
        except Exception:
            continue

    # Fallback to static config
    return dict(FALLBACK_DISPLAY)


def get_available_metrics() -> List[str]:
    """All known metric keys (from mapping or fallback)."""
    return sorted(list(_load_mapping().keys()))


def get_metric_display_name(metric: str) -> str:
    mapping = _load_mapping()
    return mapping.get(metric, metric)


def split_metrics() -> Dict[str, List[str]]:
    """Return categorized metrics for prompt/tooling convenience."""
    all_keys = get_available_metrics()
    agg_keys = set(AGGREGATED_METRICS.keys())
    aggregated = [k for k in all_keys if k in agg_keys]
    granular = [k for k in all_keys if k not in agg_keys]
    return {"aggregated": aggregated, "granular": granular}
