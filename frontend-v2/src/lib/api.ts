import type { Analysis, AsyncTask, AnalysisStatus, TrendsResponse } from './types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL as string;

function withBase(path: string): string {
  if (!API_BASE_URL) throw new Error('VITE_API_BASE_URL is not set');
  return `${API_BASE_URL}${path}`;
}

export async function startAnalysis(file: File): Promise<AsyncTask> {
  const formData = new FormData();
  formData.append('text_file', file);
  const resp = await fetch(withBase('/analyze_text/'), { method: 'POST', body: formData });
  if (!resp.ok) {
    const err = await safeJson(resp);
    throw new Error(err?.detail || 'Failed to start analysis');
  }
  return resp.json();
}

export async function getAnalysisStatus(jobId: string): Promise<AnalysisStatus> {
  const resp = await fetch(withBase(`/analysis_status/${jobId}`));
  if (!resp.ok) throw new Error(`Status check failed: ${resp.status}`);
  return resp.json();
}

export interface AnalysesResponse {
items: Analysis[];
total: number;
page: number;
  limit: number;
  has_more: boolean;
}

export async function getAnalyses(page = 1, limit = 20): Promise<AnalysesResponse> {
  const url = new URL(withBase('/analyses/'));
  url.searchParams.set('page', page.toString());
  url.searchParams.set('limit', limit.toString());
  const resp = await fetch(url.toString());
  if (!resp.ok) throw new Error('Failed to fetch analyses');
  return resp.json();
}

export async function getAnalysis(id: number, opts?: { signal?: AbortSignal }): Promise<Analysis> {
  const resp = await fetch(withBase(`/analyses/${id}`), { signal: opts?.signal });
  if (!resp.ok) throw new Error('Failed to fetch analysis');
  return resp.json();
}

export async function getTrends(metric: string, period: string): Promise<TrendsResponse> {
  const url = new URL(withBase('/api/trends'));
  url.searchParams.set('metric', metric);
  url.searchParams.set('period', period);
  const resp = await fetch(url.toString());
  if (!resp.ok) throw new Error('Failed to fetch trends');
  return resp.json();
}

export function streamInsights(body: { question: string; session_id: string }) {
  return fetch(withBase('/api/get_insights'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

async function safeJson(resp: Response): Promise<any | null> {
  try { return await resp.json(); } catch { return null; }
}

