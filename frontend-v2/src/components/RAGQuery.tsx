import { useState, useEffect, useRef, type FormEvent } from 'react';

// Define the structure of a message in the chat history
interface Message {
    sender: 'user' | 'ai';
    text: string;
    followUps?: string[];
}

// Define the structure of the API response
interface RAGAnswer {
    answer: string;
    bullets: string[];
    metrics_summary: any[];
    citations: any[];
    follow_ups: string[];
}

export function RAGQuery() {
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [query, setQuery] = useState('');
    const [history, setHistory] = useState<Message[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const chatContainerRef = useRef<HTMLDivElement>(null);

    // Effect to initialize session ID from localStorage or create a new one
    useEffect(() => {
        const storedSessionId = localStorage.getItem('rag_session_id');
        if (storedSessionId) {
            setSessionId(storedSessionId);
        } else {
            const newSessionId = `rag-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
            localStorage.setItem('rag_session_id', newSessionId);
            setSessionId(newSessionId);
        }
    }, []);

    // Effect to scroll to the bottom of the chat on new messages
    useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [history]);

    const handleSubmit = async (e: FormEvent | { currentTarget: { value: string } }, aQuery?: string) => {
        if (e) {
            e.preventDefault();
        }

        const currentQuery = aQuery || query;
        if (!currentQuery.trim() || !sessionId) return;

        setIsLoading(true);
        setHistory(prev => [...prev, { sender: 'user', text: currentQuery }]);
        setQuery('');

        try {
            const apiUrl = import.meta.env.VITE_API_BASE_URL;
            const response = await fetch(`${apiUrl}/api/get_insights`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: currentQuery,
                    session_id: sessionId,
                    // Add other filters if needed
                }),
            });

            if (!response.ok) {
                throw new Error('Failed to get insights from the AI assistant.');
            }

            const data: RAGAnswer = await response.json();
            setHistory(prev => [...prev, {
                sender: 'ai',
                text: data.answer,
                followUps: data.follow_ups,
            }]);

        } catch (error) {
            const err = error as Error;
            setHistory(prev => [...prev, { sender: 'ai', text: `Error: ${err.message}` }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleFollowUpClick = (followUp: string) => {
        handleSubmit({} as FormEvent, followUp);
    };


    return (
        <div className="rag-section">
            <h2>AI Assistant</h2>
            <div className="chat-container" ref={chatContainerRef}>
                {history.map((msg, index) => (
                    <div key={index} className={`chat-message ${msg.sender}`}>
                        <p>{msg.text}</p>
                        {msg.sender === 'ai' && msg.followUps && (
                            <div className="follow-ups">
                                {msg.followUps.map((fu, i) => (
                                    <button key={i} onClick={() => handleFollowUpClick(fu)}>
                                        {fu}
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>
                ))}
                {isLoading && <div className="chat-message ai"><p><i>AI is thinking...</i></p></div>}
            </div>
            <form onSubmit={handleSubmit} className="chat-form">
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Ask a question about the analysis..."
                    disabled={isLoading}
                />
                <button type="submit" disabled={isLoading || !query.trim()}>
                    Send
                </button>
            </form>
        </div>
    );
}
