import { createContext, useContext, useState, useCallback, type ReactNode } from 'react';
import type { Analysis } from '../lib/types';

interface AppState {
  // Upload view state
  uploadAnalysisResult: Analysis | null;
  uploadIsLoading: boolean;
  uploadError: string | null;

  // Analyses view state
  selectedAnalysisId: number | null;

  // Trends view state
  trendsMetric: string;
  trendsPeriod: 'daily' | 'weekly';
}

interface AppStateContextType {
  state: AppState;
  
  // Upload state setters
  setUploadAnalysisResult: (analysis: Analysis | null) => void;
  setUploadIsLoading: (loading: boolean) => void;
  setUploadError: (error: string | null) => void;
  
  // Analyses state setters
  setSelectedAnalysisId: (id: number | null) => void;
  
  // Trends state setters
  setTrendsMetric: (metric: string) => void;
  setTrendsPeriod: (period: 'daily' | 'weekly') => void;
}

const AppStateContext = createContext<AppStateContextType | undefined>(undefined);

const initialState: AppState = {
  uploadAnalysisResult: null,
  uploadIsLoading: false,
  uploadError: null,
  selectedAnalysisId: null,
  trendsMetric: '',
  trendsPeriod: 'daily',
};

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AppState>(initialState);

  const setUploadAnalysisResult = useCallback((analysis: Analysis | null) => {
    setState(prev => ({ ...prev, uploadAnalysisResult: analysis }));
  }, []);

  const setUploadIsLoading = useCallback((loading: boolean) => {
    setState(prev => ({ ...prev, uploadIsLoading: loading }));
  }, []);

  const setUploadError = useCallback((error: string | null) => {
    setState(prev => ({ ...prev, uploadError: error }));
  }, []);

  const setSelectedAnalysisId = useCallback((id: number | null) => {
    setState(prev => ({ ...prev, selectedAnalysisId: id }));
  }, []);

  const setTrendsMetric = useCallback((metric: string) => {
    setState(prev => ({ ...prev, trendsMetric: metric }));
  }, []);

  const setTrendsPeriod = useCallback((period: 'daily' | 'weekly') => {
    setState(prev => ({ ...prev, trendsPeriod: period }));
  }, []);

  return (
    <AppStateContext.Provider
      value={{
        state,
        setUploadAnalysisResult,
        setUploadIsLoading,
        setUploadError,
        setSelectedAnalysisId,
        setTrendsMetric,
        setTrendsPeriod,
      }}
    >
      {children}
    </AppStateContext.Provider>
  );
}

export function useAppState() {
  const context = useContext(AppStateContext);
  if (!context) {
    throw new Error('useAppState must be used within AppStateProvider');
  }
  return context;
}
