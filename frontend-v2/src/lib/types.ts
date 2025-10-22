export interface Utterance {
  date?: string;
  timestamp?: string;
  speaker: string;
  text: string;
  sa_labels?: string[];
  predictions: Record<string, number>;
  aggregated_scores: Record<string, number>;
}

export interface Analysis {
  id: number;
  source_filename: string;
  created_at?: string;
  utterances: Utterance[];
}

export type ColumnMapping = Record<string, { original_name: string; source_file: string }>

export interface TrendDataset {
  label: string;
  data: number[];
}

export interface TrendsResponse {
  labels: string[];
  datasets: TrendDataset[];
}

export interface AsyncTask {
job_id: string;
}

export interface AnalysesResponse {
  items: Analysis[];
  total: number;
  page: number;
  limit: number;
  has_more: boolean;
}

export interface AnalysisStatus {
  job_id: string;
  status: 'PENDING' | 'COMPLETED' | 'FAILED' | string;
  analysis_id?: number;
  error?: string;
}

export interface RagFinalData {
  bullets?: string[];
  metrics_summary?: any[];
  citations?: Array<{ speaker?: string; date?: string; timestamp?: string; snippet?: string }>;
  follow_ups?: string[];
}

