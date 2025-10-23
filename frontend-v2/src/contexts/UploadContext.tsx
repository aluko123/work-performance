import { createContext, useContext, useState, useCallback, type ReactNode } from 'react';
import type { Analysis } from '../lib/types';
import { startAnalysis, getAnalysisStatus, getAnalysis } from '../lib/api';

interface UploadJob {
  jobId: string;
  fileName: string;
  status: 'uploading' | 'processing' | 'completed' | 'failed';
  progress: number;
  error?: string;
  analysisId?: number;
  analysis?: Analysis;
}

interface UploadContextType {
  jobs: Map<string, UploadJob>;
  startUpload: (file: File) => Promise<string>;
  getJob: (jobId: string) => UploadJob | undefined;
  getActiveJobs: () => UploadJob[];
}

const UploadContext = createContext<UploadContextType | undefined>(undefined);

export function UploadProvider({ children }: { children: ReactNode }) {
  const [jobs, setJobs] = useState<Map<string, UploadJob>>(new Map());

  const updateJob = useCallback((jobId: string, updates: Partial<UploadJob>) => {
    setJobs(prev => {
      const newJobs = new Map(prev);
      const existing = newJobs.get(jobId);
      if (existing) {
        newJobs.set(jobId, { ...existing, ...updates });
      }
      return newJobs;
    });
  }, []);

  const startUpload = useCallback(async (file: File): Promise<string> => {
    try {
      const task = await startAnalysis(file);
      const jobId = task.job_id;

      setJobs(prev => new Map(prev).set(jobId, {
        jobId,
        fileName: file.name,
        status: 'processing',
        progress: 30,
      }));

      // Poll in background (non-blocking)
      pollJob(jobId, updateJob);
      
      return jobId;
    } catch (err: any) {
      console.error('Upload failed:', err);
      throw err;
    }
  }, [updateJob]);

  const getJob = useCallback((jobId: string) => jobs.get(jobId), [jobs]);
  
  const getActiveJobs = useCallback(() => {
    return Array.from(jobs.values()).filter(
      job => job.status === 'uploading' || job.status === 'processing'
    );
  }, [jobs]);

  return (
    <UploadContext.Provider value={{ jobs, startUpload, getJob, getActiveJobs }}>
      {children}
    </UploadContext.Provider>
  );
}

export function useUpload() {
  const context = useContext(UploadContext);
  if (!context) {
    throw new Error('useUpload must be used within UploadProvider');
  }
  return context;
}

// Background polling (non-blocking)
async function pollJob(
  jobId: string,
  updateJob: (jobId: string, updates: Partial<UploadJob>) => void
) {
  let attempts = 0;
  const maxAttempts = 120; // ~10 minutes

  while (attempts < maxAttempts) {
    try {
      const status = await getAnalysisStatus(jobId);

      if (status.status === 'COMPLETED' && status.analysis_id) {
        updateJob(jobId, { progress: 80 });
        const analysis = await getAnalysis(status.analysis_id);
        updateJob(jobId, {
          status: 'completed',
          progress: 100,
          analysisId: status.analysis_id,
          analysis,
        });
        return;
      }

      if (status.status === 'FAILED') {
        updateJob(jobId, {
          status: 'failed',
          error: status.error || 'Analysis failed',
        });
        return;
      }

      updateJob(jobId, {
        progress: 30 + (attempts / maxAttempts) * 50,
      });

      await new Promise(res => setTimeout(res, 5000));
      attempts++;
    } catch (err: any) {
      updateJob(jobId, {
        status: 'failed',
        error: err.message || 'Polling failed',
      });
      return;
    }
  }

  updateJob(jobId, {
    status: 'failed',
    error: 'Timeout waiting for analysis',
  });
}
