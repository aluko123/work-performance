import { useEffect, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { FileUpload } from '../components/FileUpload';
import { AnalysisDisplay } from '../components/AnalysisDisplay';
import { RAGQuery } from '../components/RAGQuery';
import { ProfileRadar } from '../components/ProfileRadar';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { AlertCircle, Loader2 } from 'lucide-react';
import { Alert, AlertDescription } from '../components/ui/alert';
import type { Analysis, ColumnMapping } from '../lib/types';

export default function UploadView() {
  const queryClient = useQueryClient();
  const [analysisResult, setAnalysisResult] = useState<Analysis | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [columnMapping, setColumnMapping] = useState<ColumnMapping>({});

  useEffect(() => {
    (async () => {
      try {
        const response = await fetch('/column_name_mapping.json');
        const data: ColumnMapping = await response.json();
        setColumnMapping(data);
      } catch (err) {
        console.error('Failed to fetch column mapping:', err);
        setError('Could not load column name mapping. Headers may be incorrect.');
      }
    })();
  }, []);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">AI Performance Analysis</h1>
        <p className="text-muted-foreground">
          Upload meeting transcripts for comprehensive AI-powered analysis
        </p>
      </div>

      <div className="grid gap-8 lg:grid-cols-3">
        <div className="lg:col-span-1">
          <FileUpload
            setIsLoading={setIsLoading}
            setError={setError}
            setAnalysis={setAnalysisResult}
            onCompleted={(a) => {
            setAnalysisResult(a);
            // Invalidate analyses cache to show the new analysis
            queryClient.invalidateQueries({ queryKey: ['analyses'] });
          }}
          />
        </div>

        <div className="lg:col-span-2 space-y-6">
          {isLoading && (
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center space-x-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Processing your file...</span>
                </div>
              </CardContent>
            </Card>
          )}

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {analysisResult && (
            <>
              <Card>
                <CardHeader>
                  <CardTitle>Performance Profile</CardTitle>
                  <CardDescription>
                    Overall communication and situational awareness scores
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ProfileRadar analysis={analysisResult} />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Detailed Analysis</CardTitle>
                  <CardDescription>
                    Individual utterance scores and metrics
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <AnalysisDisplay analysis={analysisResult} columnMapping={columnMapping} />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>AI Assistant</CardTitle>
                  <CardDescription>
                    Ask questions about the analysis or get insights
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <RAGQuery />
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

