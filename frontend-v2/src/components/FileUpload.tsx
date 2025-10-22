import { useState, type ChangeEvent, type FormEvent, useCallback } from "react";
import type { Analysis } from "../lib/types";
import { startAnalysis, getAnalysisStatus, getAnalysis } from "../lib/api";
import { Button } from "./ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Progress } from "./ui/progress";
import { Upload as UploadIcon, FileText, CheckCircle } from "lucide-react";
import { cn } from "../lib/utils";

export interface FileUploadProps {
    setIsLoading: (isLoading: boolean) => void;
    setError: (error: string | null) => void;
    setAnalysis: (analysis: Analysis | null) => void;
    onCompleted?: (analysis: Analysis) => void;
}

export function FileUpload({ setIsLoading, setError, setAnalysis, onCompleted }: FileUploadProps) {
    const [file, setFile] = useState<File | null>(null);
    const [dragActive, setDragActive] = useState(false);
    const [progress, setProgress] = useState(0);
    const [status, setStatus] = useState<'idle' | 'uploading' | 'processing' | 'completed'>('idle');

    const handleDrag = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
        }
    }, []);

    const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            setFile(e.target.files[0]);
        }
    };

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        if(!file) {
            setError("Please select a file first.");
            return;
        }

        setIsLoading(true);
        setError(null);
        setAnalysis(null);
        setStatus('uploading');
        setProgress(10);

        try {
            // 1) Start async job
            const task = await startAnalysis(file);
            setProgress(30);
            setStatus('processing');

            // 2) Poll for completion
            let attempts = 0;
            let analysisId: number | undefined;
            while (attempts < 120) { // up to ~10 minutes at 5s interval
                const status = await getAnalysisStatus(task.job_id);
                if (status.status === 'COMPLETED' && status.analysis_id) {
                    analysisId = status.analysis_id;
                    setProgress(80);
                    break;
                }
                if (status.status === 'FAILED') {
                    throw new Error(status.error || 'Analysis failed');
                }
                setProgress(30 + (attempts / 120) * 50);
                await new Promise(res => setTimeout(res, 5000));
                attempts += 1;
            }
            if (!analysisId) throw new Error('Timed out waiting for analysis');

            setProgress(90);
            // 3) Fetch the created analysis
            const analysis = await getAnalysis(analysisId);
            setProgress(100);
            setStatus('completed');
            setAnalysis(analysis);
            onCompleted?.(analysis);

            // Reset after success
            setTimeout(() => {
                setFile(null);
                setStatus('idle');
                setProgress(0);
            }, 2000);
        } catch (err: any) {
            setError(err.message || 'Analysis failed');
            setStatus('idle');
            setProgress(0);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <Card className="w-full max-w-2xl mx-auto">
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <UploadIcon className="h-5 w-5" />
                    Upload Meeting Transcript
                </CardTitle>
                <CardDescription>
                    Upload a transcript file (.txt, .pdf, .csv, .xlsx) for AI-powered analysis
                </CardDescription>
            </CardHeader>
            <CardContent>
                <form onSubmit={handleSubmit} className="space-y-4">
                    <div
                        className={cn(
                            "border-2 border-dashed rounded-lg p-8 text-center transition-colors",
                            dragActive ? "border-primary bg-primary/5" : "border-muted-foreground/25",
                            file && "border-primary/50"
                        )}
                        onDragEnter={handleDrag}
                        onDragLeave={handleDrag}
                        onDragOver={handleDrag}
                        onDrop={handleDrop}
                    >
                        {file ? (
                            <div className="space-y-2">
                                <FileText className="h-12 w-12 mx-auto text-primary" />
                                <p className="text-sm font-medium">{file.name}</p>
                                <p className="text-xs text-muted-foreground">
                                    {(file.size / 1024 / 1024).toFixed(2)} MB
                                </p>
                                {status === 'completed' && (
                                    <div className="flex items-center justify-center gap-1 text-green-600">
                                        <CheckCircle className="h-4 w-4" />
                                        <span className="text-sm">Analysis complete!</span>
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div className="space-y-2">
                                <UploadIcon className="h-12 w-12 mx-auto text-muted-foreground" />
                                <p className="text-sm font-medium">
                                    Drop your file here, or{" "}
                                    <label className="text-primary cursor-pointer underline">
                                        browse
                                        <input
                                            type="file"
                                            accept=".txt,.pdf,.csv,.xlsx"
                                            onChange={handleFileChange}
                                            className="hidden"
                                        />
                                    </label>
                                </p>
                                <p className="text-xs text-muted-foreground">
                                    Supports TXT, PDF, CSV, XLSX files up to 50MB
                                </p>
                            </div>
                        )}
                    </div>

                    {(status === 'uploading' || status === 'processing') && (
                        <div className="space-y-2">
                            <Progress value={progress} className="w-full" />
                            <p className="text-xs text-center text-muted-foreground">
                                {status === 'uploading' ? 'Uploading...' : 'Processing...'} {Math.round(progress)}%
                            </p>
                        </div>
                    )}

                    <Button
                        type="submit"
                        disabled={!file || status !== 'idle'}
                        className="w-full"
                    >
                        {status === 'completed' ? (
                            <>
                                <CheckCircle className="mr-2 h-4 w-4" />
                                Analysis Complete
                            </>
                        ) : status === 'processing' ? (
                            "Processing..."
                        ) : (
                            "Analyze File"
                        )}
                    </Button>
                </form>
            </CardContent>
        </Card>
    );
}
