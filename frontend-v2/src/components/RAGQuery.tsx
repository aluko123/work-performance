import { useState, useEffect, useRef, type FormEvent } from 'react';

// Define the structure of a message in the chat history
interface Message {
    sender: 'user' | 'ai';
    text: string;
    followUps?: string[];
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

    const submitQuery = async (queryToSubmit: string) => {
        if (!queryToSubmit.trim() || !sessionId) return;

        setIsLoading(true);
        setHistory(prev => [...prev, { sender: 'user', text: queryToSubmit }]);
        setQuery(''); // Clear input field

        // Add an empty AI message placeholder
        setHistory(prev => [...prev, { sender: 'ai', text: '' }]);

        try {
            const apiUrl = import.meta.env.VITE_API_BASE_URL;
            const response = await fetch(`${apiUrl}/api/get_insights`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: queryToSubmit,
                    session_id: sessionId,
                }),
            });

            if (!response.ok) {
                throw new Error('Failed to get insights from the AI assistant.');
            }

            if (!response.body) {
                throw new Error("Response body is empty.");
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    break;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || ''; // Keep the last partial line in the buffer

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const jsonString = line.substring(6);
                        if (jsonString.trim() === '[DONE]') {
                            break;
                        }
                        try {
                            const data = JSON.parse(jsonString);
                            if (data.answer_token) {
                                setHistory(prev => {
                                    const newHistory = [...prev];
                                    const lastMessage = newHistory[newHistory.length - 1];
                                    if (lastMessage && lastMessage.sender === 'ai') {
                                        lastMessage.text += data.answer_token;
                                    }
                                    return newHistory;
                                });
                            }
                            if (data.follow_ups) {
                                setHistory(prev => {
                                    const newHistory = [...prev];
                                    const lastMessage = newHistory[newHistory.length - 1];
                                    if (lastMessage && lastMessage.sender === 'ai') {
                                        lastMessage.followUps = data.follow_ups;
                                    }
                                    return newHistory;
                                });
                            }
                        } catch (e) {
                            console.error("Failed to parse stream chunk:", jsonString);
                        }
                    }
                }
            }

        } catch (error) {
            const err = error as Error;
            setHistory(prev => {
                const newHistory = [...prev];
                const lastMessage = newHistory[newHistory.length - 1];
                if (lastMessage && lastMessage.sender === 'ai' && lastMessage.text === '') {
                    lastMessage.text = `Error: ${err.message}`;
                } else {
                    newHistory.push({ sender: 'ai', text: `Error: ${err.message}` });
                }
                return newHistory;
            });
        } finally {
            setIsLoading(false);
        }
    };

    const handleSubmit = (e: FormEvent) => {
        e.preventDefault();
        submitQuery(query);
    };

    const handleFollowUpClick = (followUp: string) => {
        submitQuery(followUp);
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
                {isLoading && history[history.length -1]?.sender === 'user' && <div className="chat-message ai"><p><i>AI is thinking...</i></p></div>}
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