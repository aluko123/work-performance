import {useState, useEffect } from 'react';
import './App.css';
import { FileUpload } from './components/FileUpload';
import { RAGQuery } from './components/RAGQuery';
import { AnalysisDisplay } from './components/AnalysisDisplay';
//import { RAGQuery } from './components/RAGQuery';

//analysis type
export interface Analysis {
  id: number;
  source_filename: string;
  utterances: any[];   
}


export type ColumnMapping = Record<string, { original_name: string; source_file: string; }>;

function App() {
  const [analysisResult, setAnalysisResult] = useState<Analysis | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [columnMapping, setColumnMapping] = useState<ColumnMapping>({});

  //useEffect to fetch mapping file when app loads
  useEffect(() => {
    const fetchMapping = async () => {
      try {
        const response = await fetch('/column_name_mapping.json');
        const data: ColumnMapping = await response.json();
        setColumnMapping(data);
      } catch (err) {
        console.error("Failed to fetch column mapping:", err);
        setError("Could not load column name mapping. Headers may be incorrect.");
      } 
    };
    fetchMapping();
  }, []); 

  
  return (
    <div className="app-container">
      <header>
        <h1>AI Performance Analysis</h1>
      </header>
      <main>
        <FileUpload
        setIsLoading={setIsLoading}
        setError={setError}
        setAnalysis={setAnalysisResult}
        />
        {isLoading && <div className='loading-indicator'>Processing...</div>}
        {error && <div className='error-message'>{error}</div>}
        {analysisResult && <AnalysisDisplay analysis={analysisResult} columnMapping={columnMapping} />}
        {analysisResult && <RAGQuery />}
      </main>
    </div>
  );
}

export default App;