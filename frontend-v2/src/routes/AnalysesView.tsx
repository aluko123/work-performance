import { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { AnalysesList } from '../components/AnalysesList';
import { AnalysisDisplay } from '../components/AnalysisDisplay';
import { RAGQuery } from '../components/RAGQuery';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { getAnalysis } from '../lib/api';
import { useAppState } from '../contexts/AppStateContext';
import type { Analysis, ColumnMapping } from '../lib/types';

export default function AnalysesView() {
  const { state, setSelectedAnalysisId } = useAppState();
  const [searchParams, setSearchParams] = useSearchParams();
  
  // Use URL param if present, otherwise fall back to context state
  const urlId = searchParams.get('id') ? Number(searchParams.get('id')) : null;
  const selectedId = urlId ?? state.selectedAnalysisId;
  
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [columnMapping, setColumnMapping] = useState<ColumnMapping>({});

  // Sync URL with context state
  useEffect(() => {
    if (selectedId !== null && !urlId) {
      setSearchParams({ id: selectedId.toString() });
    }
  }, [selectedId, urlId, setSearchParams]);

  const handleSelect = (id: number) => {
    setSearchParams({ id: id.toString() });
    setSelectedAnalysisId(id);
  };

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
    
    setAnalysis(null); // Unmount heavy table immediately
    const ac = new AbortController();
    let mounted = true;
    
    (async () => {
      setLoading(true); setError(null);
      try {
        const data = await getAnalysis(selectedId, { signal: ac.signal });
        if (mounted) setAnalysis(data);
      } catch (e: any) {
        if (mounted && e.name !== 'AbortError') setError(e.message);
      } finally { 
        if (mounted) setLoading(false); 
      }
    })();
    
    return () => { 
      mounted = false; 
      ac.abort();
    };
  }, [selectedId]);

  return (
    <div className="app-container" style={{ display: 'grid', gridTemplateColumns: '340px 1fr', gap: '1rem' }}>
      <aside className="sidebar" style={{ border: '1px solid #333', borderRadius: 8, padding: '1rem' }}>
        <AnalysesList selectedId={selectedId} onSelect={handleSelect} />
      </aside>
      <main className="main-content">
        {loading && <div className='loading-indicator'>Loading...</div>}
        {error && <div className='error-message'>{error}</div>}
        {analysis && (
          <>
            <h2>Details for {analysis.source_filename}</h2>
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
