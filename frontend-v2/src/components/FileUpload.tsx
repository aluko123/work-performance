import { useState, type ChangeEvent, type FormEvent } from "react";
import type { Analysis } from "../App";

export interface FileUploadProps {
    setIsLoading: (isLoading: boolean) => void;
    setError: (error: string | null) => void;
    setAnalysis: (analysis: Analysis | null) => void;
}

export function FileUpload({ setIsLoading, setError, setAnalysis }: FileUploadProps) {
    const [file, setFile] = useState<File | null>(null);

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

        const formData = new FormData();
        formData.append("text_file", file);

        try {

            const apiUrl = import.meta.env.VITE_API_BASE_URL;

            const response = await fetch(`${apiUrl}/analyze_text/`, {
                method:'POST',
                body: formData,
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Analysis failed");
            }

            const data: Analysis = await response.json();
            setAnalysis(data);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="upload-section">
            <h2>Upload Meeting Transcript</h2>
            <form onSubmit={handleSubmit}>
                <input type="file" onChange={handleFileChange} />
                <button type="submit" disabled={!file}>Analyze File</button>
            </form>
        </div>
    );
}