"""
Chart feature configuration
Loads from environment variables with sensible defaults
"""

import os
from typing import List, Dict


def str_to_bool(value: str) -> bool:
    return value.lower() in ('true', '1', 'yes', 'on')


def str_to_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(',')]


# Chart Feature Flag
ENABLE_CHARTS = str_to_bool(os.getenv('ENABLE_CHARTS', 'true'))

# Chart Limits
MAX_CHARTS_PER_RESPONSE = int(os.getenv('MAX_CHARTS_PER_RESPONSE', '2'))
CHART_MAX_DATA_POINTS = int(os.getenv('CHART_MAX_DATA_POINTS', '50'))
CHART_CACHE_SIZE = int(os.getenv('CHART_CACHE_SIZE', '100'))

# Chart Styling
CHART_DEFAULT_COLORS = str_to_list(
    os.getenv('CHART_DEFAULT_COLORS', '#8884d8,#82ca9d,#ffc658,#ff7c7c,#8dd1e1')
)

# Aggregated metrics (high-level rollups)
AGGREGATED_METRICS = {
    'SAFETY_Score': 'Safety Performance',
    'QUALITY_Score': 'Quality Performance',
    'DELIVERY_Score': 'Delivery Performance',
    'COST_Score': 'Cost Performance',
    'PEOPLE_Score': 'People Performance',
    'Total_Comm_Score': 'Communication Score',
    'Total_Deviation_Score': 'Deviation Management Score',
    'Feedback_Tier1_Score': 'Feedback Quality (Tier 1)',
    'Feedback_Tier2_Score': 'Feedback Quality (Tier 2)',
}

# Granular metrics (specific behaviors) - Tier 1 priority
GRANULAR_METRICS = {
    # Communication
    'comm_Pausing': 'Pausing',
    'comm_Verbal_Affirmation': 'Verbal Affirmation',
    'comm_Clarifying_Questions': 'Clarifying Questions',
    'comm_Probing_Questions': 'Probing Questions',
    'comm_Open_Ended_Questions': 'Open-Ended Questions',
    'comm_Coaching_Questions': 'Coaching Questions',
    'comm_Acknowledgment_of_Emotions': 'Acknowledgment of Emotions',
    'comm_Summary_Statements': 'Summary Statements',
    
    # Feedback
    'feedback_Timely': 'Timely Feedback',
    'feedback_Neutral___Specific': 'Neutral & Specific Feedback',
    'feedback_Impact___Emotion': 'Impact & Emotion in Feedback',
    'feedback_Action_Oriented': 'Action-Oriented Feedback',
    'feedback_Clarity_of_Situation': 'Clarity of Situation',
    
    # Safety (SQDCP)
    'sqdcp_Hazard_Identification': 'Hazard Identification',
    'sqdcp_Safety_Coaching_Language': 'Safety Coaching Language',
    'sqdcp_Emotional_Awareness': 'Emotional Awareness',
    'sqdcp_PPE___Compliance_Visibility': 'PPE & Compliance',
    
    # Quality
    'sqdcp_Clear_Defect_Description': 'Clear Defect Description',
    'sqdcp_Root_Cause_Exploration': 'Root Cause Exploration',
    'sqdcp_Trend_Recognition': 'Trend Recognition',
    
    # Deviation Management
    'deviation_Current_Target': 'Current Target',
    'deviation_Deviation_from_Target': 'Deviation from Target',
    'deviation_Containment___Short_Term_Fix': 'Containment & Short-Term Fix',
    'deviation_Preventive_Action___System_Fix': 'Preventive Action',
    'deviation_Trend_Recognition': 'Deviation Trend Recognition',
    'deviation_Define_Next_Actions': 'Next Actions Defined',
}

# Combined display names
METRIC_DISPLAY_NAMES = {**AGGREGATED_METRICS, **GRANULAR_METRICS}

# Metric descriptions for LLM context
METRIC_DESCRIPTIONS: Dict[str, str] = {
    'SAFETY_Score': 'Overall safety performance including hazard identification, PPE compliance, and safety coaching',
    'QUALITY_Score': 'Overall quality including defect description, root cause analysis, and trend recognition',
    'DELIVERY_Score': 'Overall delivery performance including deviation management and containment actions',
    'COST_Score': 'Cost awareness including waste, rework, and efficiency',
    'PEOPLE_Score': 'People management including feedback, participation, and emotional intelligence',
    'Total_Comm_Score': 'Overall communication effectiveness across all behaviors',
    'Total_Deviation_Score': 'Overall deviation management and problem-solving effectiveness',
}
