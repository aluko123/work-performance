import { useState, type ChangeEvent, type FormEvent, useCallback, useEffect } from "react";
import type { Analysis } from "../lib/types";
import { useUpload } from "../contexts/UploadContext";
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
    const { startUpload, getJob, getActiveJobs } = useUpload();
    const [file, setFile] = useState<File | null>(null);
    const [dragActive, setDragActive] = useState(false);
    const [currentJobId, setCurrentJobId] = useState<string | null>(null);
    
    const activeJobs = getActiveJobs();
    const currentJob = currentJobId ? getJob(currentJobId) : undefined;

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

    useEffect(() => {
        if (!currentJob) {
            setIsLoading(false);
            return;
        }

        setIsLoading(currentJob.status === 'processing' || currentJob.status === 'uploading');

        if (currentJob.status === 'completed' && currentJob.analysis) {
            setAnalysis(currentJob.analysis);
            onCompleted?.(currentJob.analysis);
            setTimeout(() => {
                setFile(null);
                setCurrentJobId(null);
            }, 2000);
        }

        if (currentJob.status === 'failed') {
            setError(currentJob.error || 'Upload failed');
            setIsLoading(false);
        }
    }, [currentJob, setIsLoading, setError, setAnalysis, onCompleted]);

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        if(!file) {
            setError("Please select a file first.");
            return;
        }

        setIsLoading(true);
        setError(null);
        setAnalysis(null);

        try {
            const jobId = await startUpload(file);
            setCurrentJobId(jobId);
        } catch (err: any) {
            setError(err.message || 'Upload failed');
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
                                {currentJob?.status === 'completed' && (
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

                    {currentJob && (currentJob.status === 'uploading' || currentJob.status === 'processing') && (
                        <div className="space-y-2">
                            <Progress value={currentJob.progress} className="w-full" />
                            <p className="text-xs text-center text-muted-foreground">
                                {currentJob.status === 'uploading' ? 'Uploading...' : 'Processing...'} {Math.round(currentJob.progress)}%
                            </p>
                            <p className="text-xs text-center text-muted-foreground italic">
                                {currentJob.fileName}
                            </p>
                        </div>
                    )}
                    
                    {activeJobs.length > 1 && (
                        <div className="text-xs text-muted-foreground text-center">
                            {activeJobs.length} uploads in progress
                        </div>
                    )}

                    <Button
                        type="submit"
                        disabled={!file || (currentJob !== null && ['uploading', 'processing'].includes(currentJob?.status || ''))}
                        className="w-full"
                    >
                        {currentJob?.status === 'completed' ? (
                            <>
                                <CheckCircle className="mr-2 h-4 w-4" />
                                Analysis Complete
                            </>
                        ) : currentJob?.status === 'processing' ? (
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
