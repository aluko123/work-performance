from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import torch
import torch.nn as nn
from transformers import BertModel

# --- Pydantic Models for API I/O ---

class Utterance(BaseModel):
    id: int
    analysis_id: int
    date: str | None
    timestamp: str | None
    speaker: str | None
    text: str | None
    predictions: Dict[str, Any]
    aggregated_scores: Dict[str, Any]
    sa_labels: List[str] | None

    class Config:
        from_attributes = True

class Analysis(BaseModel):
    id: int
    source_filename: str
    created_at: datetime
    utterances: List[Utterance] = Field(default_factory=list)
    job_id: Optional[str] = None

    class Config:
        from_attributes = True

class AnalysesResponse(BaseModel):
    items: List[Analysis]
    total: int
    page: int
    limit: int
    has_more: bool

class AsyncTask(BaseModel):
    job_id: str

class AnalysisResult(BaseModel):
    Date: str | None
    Timestamp: str | None
    Speaker: str | None
    Utterance: str | None
    predictions: Dict[str, int]
    aggregated_scores: Dict[str, int]

class AnalysisResponse(BaseModel):
    results: List[Dict[str, Any]]

class TrendDataset(BaseModel):
    label: str
    data: List[float | None]

class TrendsResponse(BaseModel):
    labels: List[str]
    datasets: List[TrendDataset]

# --- Model Definition ---
class MultiTaskBertModel(nn.Module):
    def __init__(self, n_classes_dict, bert_model):
        super(MultiTaskBertModel, self).__init__()
        self.bert = bert_model
        self.classifiers = nn.ModuleDict({
            task_name: nn.Linear(self.bert.config.hidden_size, num_labels)
            for task_name, num_labels in n_classes_dict.items()
        })

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        bert_output = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = bert_output.pooler_output
        outputs = {
            task_name: classifier(pooled_output)
            for task_name, classifier in self.classifiers.items()
        }
        return outputs

class RAGQuery(BaseModel):
    question: str
    session_id: Optional[str] = None
    speaker: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    top_k: Optional[int] = 8


class RAGCitation(BaseModel):
    source_id: int
    speaker: Optional[str] = None
    date: Optional[str] = None
    timestamp: Optional[str] = None
    snippet: Optional[str] = None


class ChartDataset(BaseModel):
    name: str
    data: List[float]


class ChartConfig(BaseModel):
    xAxisLabel: str
    yAxisLabel: str
    title: str
    colors: Optional[List[str]] = None


class Chart(BaseModel):
    type: str  # "line", "bar", "grouped_bar"
    data: Dict[str, Any]  # {"labels": [...], "datasets": [...]}
    config: ChartConfig


class RAGAnswer(BaseModel):
    answer: str
    bullets: List[str] = []
    metrics_summary: List[Dict[str, Any]] = []
    citations: List[RAGCitation] = []
    follow_ups: List[str] = []
    charts: List[Chart] = []  # NEW: Chart data
    metadata: Dict[str, Any] = {}
