import { useState, useEffect, useRef, type FormEvent } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card } from './ui/card';
import { Badge } from './ui/badge';
import { Send, Bot, User, Loader2 } from 'lucide-react';
import { cn } from '../lib/utils';
import { ChartRenderer } from './ChartRenderer';

// Define the structure of a message in the chat history
interface ChartDataset {
    name: string;
    data: number[];
}

interface ChartConfig {
    xAxisLabel: string;
    yAxisLabel: string;
    title: string;
    colors?: string[];
}

interface Chart {
    type: 'line' | 'bar' | 'grouped_bar';
    data: {
        labels: string[];
        datasets: ChartDataset[];
    };
    config: ChartConfig;
}

interface Message {
    sender: 'user' | 'ai';
    text: string;
    followUps?: string[];
    bullets?: string[];
    citations?: Array<{ speaker?: string; date?: string; timestamp?: string; snippet?: string }>;
    charts?: Chart[];
}

interface RAGQueryProps {
    sessionId?: string;
}

export function RAGQuery({ sessionId: propSessionId }: RAGQueryProps = {}) {
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [query, setQuery] = useState('');
    const [history, setHistory] = useState<Message[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const chatContainerRef = useRef<HTMLDivElement>(null);

    // Effect to initialize session ID
    useEffect(() => {
        if (propSessionId) {
            setSessionId(propSessionId);
        } else {
            const storedSessionId = localStorage.getItem('rag_session_id');
            if (storedSessionId) {
                setSessionId(storedSessionId);
            } else {
                const newSessionId = `rag-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
                localStorage.setItem('rag_session_id', newSessionId);
                setSessionId(newSessionId);
            }
        }
    }, [propSessionId]);

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
            const response = await fetch(`${apiUrl}/api/chat`, {
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
                            
                            // Handle answer tokens (streaming text)
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
                            
                            // Handle final complete data (includes bullets, citations, charts, follow-ups)
                            if (data.answer || data.bullets || data.citations || data.charts || data.follow_ups) {
                                setHistory(prev => {
                                    const newHistory = [...prev];
                                    const lastMessage = newHistory[newHistory.length - 1];
                                    if (lastMessage && lastMessage.sender === 'ai') {
                                        if (data.answer && !lastMessage.text) lastMessage.text = data.answer;
                                        if (data.bullets && data.bullets.length > 0) lastMessage.bullets = data.bullets;
                                        if (data.citations && data.citations.length > 0) lastMessage.citations = data.citations;
                                        if (data.charts && data.charts.length > 0) {
                                            console.log('📊 Charts received:', data.charts.length, data.charts);
                                            lastMessage.charts = data.charts;
                                        }
                                        if (data.follow_ups && data.follow_ups.length > 0) lastMessage.followUps = data.follow_ups;
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

    const extractCitationQuote = (snippet?: string): string => {
        if (!snippet) return '';
        const firstLine = snippet.split('\n')[0] || snippet;
        
        const saidMatch = firstLine.match(/said:\s*['"']?(.*?)['"']?\s*$/i);
        if (saidMatch?.[1]) return saidMatch[1].trim();
        
        const stripped = firstLine.replace(/^On .*? said:\s*/i, '').trim();
        return stripped.replace(/^['"""]|['"""]$/g, '').trim();
    };

    return (
        <div className="flex flex-col h-[600px]">
            <div className="flex-1 overflow-y-auto p-4 space-y-4" ref={chatContainerRef}>
                {history.length === 0 && (
                    <div className="text-center text-muted-foreground py-12">
                        <Bot className="h-16 w-16 mx-auto mb-4 opacity-30" />
                        <h3 className="text-lg font-semibold mb-2 text-foreground">Ask About Your Analysis</h3>
                        <p className="text-sm mb-4">I can help you explore trends, compare speakers, and find insights</p>
                        <div className="max-w-md mx-auto space-y-2 text-left text-xs">
                            <p className="flex items-start gap-2">
                                <span className="text-primary">•</span>
                                <span>"Has safety improved over time?"</span>
                            </p>
                            <p className="flex items-start gap-2">
                                <span className="text-primary">•</span>
                                <span>"Compare Jordan and Tasha on communication"</span>
                            </p>
                            <p className="flex items-start gap-2">
                                <span className="text-primary">•</span>
                                <span>"What did people say about quality?"</span>
                            </p>
                        </div>
                    </div>
                )}
                {history.map((msg, index) => (
                    <div
                        key={index}
                        className={cn(
                            "flex gap-3",
                            msg.sender === 'user' ? "justify-end" : "justify-start"
                        )}
                    >
                        {msg.sender === 'ai' && (
                            <div className="flex-shrink-0">
                                <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center">
                                    <Bot className="h-4 w-4 text-primary-foreground" />
                                </div>
                            </div>
                        )}
                        <div
                            className={cn(
                                "max-w-[80%] rounded-lg px-4 py-2",
                                msg.sender === 'user'
                                    ? "bg-primary text-primary-foreground"
                                    : "bg-muted"
                            )}
                        >
                            <div className="whitespace-pre-wrap prose prose-sm max-w-none dark:prose-invert">
                                {msg.text.split('\n').map((line, i) => {
                                    // Simple markdown parsing for bold
                                    const parsedLine = line
                                        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                                        .replace(/\*(.+?)\*/g, '<em>$1</em>');
                                    return (
                                        <p key={i} dangerouslySetInnerHTML={{ __html: parsedLine }} />
                                    );
                                })}
                            </div>

                            {msg.sender === 'ai' && Array.isArray(msg.bullets) && msg.bullets.length > 0 && (
                                <ul className="mt-2 space-y-1">
                                    {msg.bullets.map((b, i) => (
                                        <li key={i} className="text-sm flex items-start gap-2">
                                            <span className="text-primary mt-1">•</span>
                                            <span>{b}</span>
                                        </li>
                                    ))}
                                </ul>
                            )}

                            {msg.sender === 'ai' && Array.isArray(msg.charts) && msg.charts.length > 0 && (
                                <div className="mt-4 space-y-3">
                                    <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Visualizations</h4>
                                    {msg.charts.map((chart, i) => (
                                        <ChartRenderer key={i} chart={chart} />
                                    ))}
                                </div>
                            )}

                            {msg.sender === 'ai' && Array.isArray(msg.citations) && msg.citations.length > 0 && (
                                <div className="mt-4 space-y-2">
                                    <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Sources</h4>
                                    {msg.citations.map((c, i) => {
                                        const quote = extractCitationQuote(c.snippet);
                                        return (
                                            <Card key={i} className="p-3 bg-secondary/30">
                                                <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-muted-foreground mb-2">
                                                    {[c.speaker, c.date, c.timestamp].filter(Boolean).map((item, idx) => (
                                                        <Badge key={idx} variant="secondary" className="text-[11px] font-normal">
                                                            {item}
                                                        </Badge>
                                                    ))}
                                                </div>
                                                {quote && (
                                                    <p className="text-sm italic text-foreground/90 leading-relaxed">
                                                        "{quote}"
                                                    </p>
                                                )}
                                            </Card>
                                        );
                                    })}
                                </div>
                            )}

                            {msg.sender === 'ai' && msg.followUps && msg.followUps.length > 0 && (
                                <div className="mt-3 flex flex-wrap gap-2">
                                    {msg.followUps.map((fu, i) => (
                                        <Button
                                            key={i}
                                            variant="outline"
                                            size="sm"
                                            onClick={() => handleFollowUpClick(fu)}
                                            className="text-xs"
                                        >
                                            {fu}
                                        </Button>
                                    ))}
                                </div>
                            )}
                        </div>
                        {msg.sender === 'user' && (
                            <div className="flex-shrink-0">
                                <div className="w-8 h-8 bg-secondary rounded-full flex items-center justify-center">
                                    <User className="h-4 w-4 text-secondary-foreground" />
                                </div>
                            </div>
                        )}
                    </div>
                ))}

                {isLoading && history.length > 0 && history[history.length - 1]?.sender === 'user' && (
                    <div className="flex gap-3 justify-start">
                        <div className="flex-shrink-0">
                            <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center">
                                <Bot className="h-4 w-4 text-primary-foreground" />
                            </div>
                        </div>
                        <div className="bg-muted rounded-lg px-4 py-2">
                            <div className="flex items-center gap-2">
                                <Loader2 className="h-4 w-4 animate-spin" />
                                <span className="text-sm text-muted-foreground">AI is thinking...</span>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            <div className="border-t p-4">
                <form onSubmit={handleSubmit} className="flex gap-2">
                    <Input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Ask a question about the analysis..."
                        disabled={isLoading}
                        className="flex-1"
                    />
                    <Button
                        type="submit"
                        disabled={isLoading || !query.trim()}
                        size="icon"
                    >
                        <Send className="h-4 w-4" />
                    </Button>
                </form>
            </div>
        </div>
    );
}
