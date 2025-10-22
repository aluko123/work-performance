import { useEffect, useState } from 'react';
import { AnalysesList } from '../components/AnalysesList';
import { AnalysisDisplay } from '../components/AnalysisDisplay';
import { ProfileRadar } from '../components/ProfileRadar';
import { RAGQuery } from '../components/RAGQuery';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { getAnalysis } from '../lib/api';
import type { Analysis, ColumnMapping } from '../lib/types';

export default function AnalysesView() {
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [columnMapping, setColumnMapping] = useState<ColumnMapping>({});

  useEffect(() => {
    (async () => {
      try {
        const resp = await fetch('/column_name_mapping.json');
        setColumnMapping(await resp.json());
      } catch (e) {
        // non-fatal
      }
    })();
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    let mounted = true;
    (async () => {
      setLoading(true); setError(null);
      try {
        const data = await getAnalysis(selectedId);
        if (mounted) setAnalysis(data);
      } catch (e: any) {
        if (mounted) setError(e.message);
      } finally { if (mounted) setLoading(false); }
    })();
    return () => { mounted = false };
  }, [selectedId]);

  return (
    <div className="app-container" style={{ display: 'grid', gridTemplateColumns: '340px 1fr', gap: '1rem' }}>
      <aside className="sidebar" style={{ border: '1px solid #333', borderRadius: 8, padding: '1rem' }}>
        <AnalysesList selectedId={selectedId} onSelect={setSelectedId} />
      </aside>
      <main className="main-content">
        {loading && <div className='loading-indicator'>Loading...</div>}
        {error && <div className='error-message'>{error}</div>}
        {analysis && (
          <>
            <h2>Details for {analysis.source_filename}</h2>
            <ProfileRadar analysis={analysis} />
            <AnalysisDisplay analysis={analysis} columnMapping={columnMapping} />

            <Card className="mt-6">
              <CardHeader>
                <CardTitle>AI Assistant</CardTitle>
                <CardDescription>
                  Ask questions about this meeting analysis or get insights
                </CardDescription>
              </CardHeader>
              <CardContent>
                <RAGQuery />
              </CardContent>
            </Card>
          </>
        )}
        {!analysis && !loading && (
          <div>Select an analysis from the left.</div>
        )}
      </main>
    </div>
  );
}
