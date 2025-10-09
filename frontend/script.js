document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const analyzeButton = document.getElementById('analyzeButton');
    const fileInput = document.getElementById('fileInput');
    const statusDiv = document.getElementById('status');
    const analysesList = document.getElementById('analysesList');
    const mainTitle = document.getElementById('main-title');
    
    // Profile Chart
    const profileChartContainer = document.getElementById('profileChartContainer');
    const performanceChartCanvas = document.getElementById('performanceChart');
    let performanceChart;

    // Trends Chart
    const trendsChartContainer = document.getElementById('trendsChartContainer');
    const trendsChartCanvas = document.getElementById('trendsChart');
    const trendsMetricSelector = document.getElementById('trendsMetricSelector');
    const trendsPeriodSelector = document.getElementById('trendsPeriodSelector');
    let trendsChart;

    // Results Table
    const resultsTable = document.getElementById('resultsTable');
    const resultsThead = document.getElementById('resultsThead');
    const resultsTbody = document.getElementById('resultsTbody');

    //RAG Insights
    const ragQueryInput = document.getElementById('ragQueryInput');
    const ragQueryButton = document.getElementById('ragQueryButton');
    const ragHistoryContainer = document.getElementById('ragHistoryContainer'); // New container for chat history


    // --- Constants ---
    const API_BASE_URL = 'http://localhost:8000';
    let columnMapping = {};
    let ragSessionId = null;

    // --- Initialization ---
    async function initializeApp() {
        await loadColumnMapping();
        const analyses = await loadAnalyses();
        if (analyses && analyses.length > 0) {
            loadAnalysisDetails(analyses[0].id);
        }
        populateTrendsMetricSelector();
        renderTrendsChart();
        initializeRagSession();
    }
    initializeApp();

    // --- Event Listeners ---
    analyzeButton.addEventListener('click', handleAnalyzeClick);
    trendsMetricSelector.addEventListener('change', renderTrendsChart);
    trendsPeriodSelector.addEventListener('change', renderTrendsChart);
    ragQueryButton.addEventListener('click', () => handleRagQuery(ragQueryInput.value));
    ragQueryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            handleRagQuery(ragQueryInput.value);
        }
    });

    // --- Core Functions ---

    function initializeRagSession() {
        ragSessionId = localStorage.getItem('rag_session_id');
        if (!ragSessionId) {
            ragSessionId = `rag-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
            localStorage.setItem('rag_session_id', ragSessionId);
        }
    }

    async function loadColumnMapping() {
        try {
            const response = await fetch('./column_name_mapping.json');
            if (!response.ok) throw new Error('Network response was not ok');
            columnMapping = await response.json();
        } catch (error) {
            console.error('Error loading column mapping:', error);
            updateStatus('Error: Could not load column name mapping.', 'red');
        }
    }

    function getDisplayName(rawName) {
        if (columnMapping[rawName] && columnMapping[rawName].original_name) {
            return columnMapping[rawName].original_name;
        }
        return (rawName.charAt(0).toUpperCase() + rawName.slice(1)).replace(/_/g, ' ');
    }

    async function loadAnalyses() {
        try {
            const response = await fetch(`${API_BASE_URL}/analyses/`);
            const data = await response.json();
            analysesList.innerHTML = '';
            data.forEach(analysis => {
                const li = document.createElement('li');
                li.textContent = `${analysis.source_filename} - ${new Date(analysis.created_at).toLocaleString()}`;
                li.dataset.id = analysis.id;
                li.addEventListener('click', () => loadAnalysisDetails(analysis.id));
                analysesList.appendChild(li);
            });
            return data; // Return data for chaining
        } catch (error) {
            console.error('Error loading analyses:', error);
            return []; // Return empty array on error
        }
    }

    function handleAnalyzeClick() {
        const file = fileInput.files[0];
        if (!file) {
            updateStatus('Please select a file first.', 'orange');
            return;
        }

        updateStatus('Uploading file and starting analysis...', 'blue');
        analyzeButton.disabled = true;
        fileInput.disabled = true;

        const formData = new FormData();
        formData.append('text_file', file);

        // 1. Start the analysis job
        fetch(`${API_BASE_URL}/analyze_text/`, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.detail || 'Failed to start analysis job.') });
            }
            return response.json();
        })
        .then(data => {
            if (data.job_id) {
                updateStatus('Analysis in progress. Please wait...', 'blue');
                // 2. Start polling for the result
                pollAnalysisStatus(data.job_id);
            } else {
                throw new Error('Did not receive a job ID from the server.');
            }
        })
        .catch(error => {
            console.error('Error during analysis initiation:', error);
            updateStatus(`Error: ${error.message}`, 'red');
            analyzeButton.disabled = false;
            fileInput.disabled = false;
        });
    }

    function pollAnalysisStatus(jobId) {
        const intervalId = setInterval(() => {
            fetch(`${API_BASE_URL}/analysis_status/${jobId}`)
                .then(response => {
                    if (!response.ok) {
                        // If the server returns a 404 or other error, stop polling.
                        throw new Error(`Server returned an error: ${response.status}`);
                    }
                    return response.json();
                })
                .then(async (data) => { // Make callback async
                    if (data.status === 'COMPLETED') {
                        // 3. Job is done
                        clearInterval(intervalId);
                        updateStatus('Analysis complete.', 'green');
                        
                        // Refresh the list, then load the new item
                        await loadAnalyses(); 
                        if (data.analysis_id) {
                            loadAnalysisDetails(data.analysis_id);
                        }
                        renderTrendsChart(); // Refresh trends chart

                        // Re-enable UI
                        analyzeButton.disabled = false;
                        fileInput.disabled = false;
                        fileInput.value = ''; // Reset file input

                    } else if (data.status === 'FAILED') {
                        // 4. Job failed
                        clearInterval(intervalId);
                        updateStatus(`Analysis failed: ${data.error || 'Unknown error'}`, 'red');
                        analyzeButton.disabled = false;
                        fileInput.disabled = false;

                    } else {
                        // 5. Job is still processing
                        updateStatus(`Analysis in progress (${data.status || '...'})`, 'blue');
                    }
                })
                .catch(error => {
                    // 6. Polling failed
                    clearInterval(intervalId);
                    console.error('Error during status polling:', error);
                    updateStatus(`Error checking analysis status: ${error.message}`, 'red');
                    analyzeButton.disabled = false;
                    fileInput.disabled = false;
                });
        }, 5000); // Poll every 5 seconds
    }


    //handle RAG query with streaming
    async function handleRagQuery(question) {
        if (!question.trim()) {
            updateStatus('Please enter a question first.', 'orange');
            return;
        }

        ragQueryInput.value = ''; // Clear input
        ragQueryButton.disabled = true;

        // 1. Display user's message
        const userMessageDiv = document.createElement('div');
        userMessageDiv.className = 'chat-message user';
        userMessageDiv.textContent = question;
        ragHistoryContainer.appendChild(userMessageDiv);

        // 2. Create placeholder for AI response
        const aiMessageDiv = document.createElement('div');
        aiMessageDiv.className = 'chat-message ai';
        const answerP = document.createElement('p');
        answerP.textContent = 'Thinking...';
        aiMessageDiv.appendChild(answerP);
        ragHistoryContainer.appendChild(aiMessageDiv);
        ragHistoryContainer.scrollTop = ragHistoryContainer.scrollHeight; // Scroll down

        try {
            const response = await fetch(`${API_BASE_URL}/api/get_insights`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: question, session_id: ragSessionId }),
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Failed to get insights');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let finalData = null;

            answerP.textContent = ''; // Clear 'Thinking...'

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const jsonString = line.substring(6);
                        if (jsonString.trim() === '[DONE]') continue;
                        try {
                            const chunk = JSON.parse(jsonString);
                            if (chunk.answer_token) {
                                answerP.textContent += chunk.answer_token;
                            } else {
                                finalData = chunk;
                            }
                        } catch (e) {
                            console.error('Failed to parse stream chunk:', jsonString);
                        }
                    }
                }
            }

            // 3. Render final data (bullets, citations, follow-ups)
            if (finalData) {
                renderRagFinalData(aiMessageDiv, finalData);
            }

        } catch (error) {
            answerP.textContent = 'An error occurred: ' + error.message;
            answerP.style.color = 'red';
        } finally {
            ragQueryButton.disabled = false;
            ragHistoryContainer.scrollTop = ragHistoryContainer.scrollHeight;
        }
    }

    function renderRagFinalData(container, data) {
        // Bullets
        if (Array.isArray(data.bullets) && data.bullets.length > 0) {
            const ul = document.createElement('ul');
            data.bullets.forEach(b => {
                const li = document.createElement('li');
                li.textContent = b;
                ul.appendChild(li);
            });
            container.appendChild(ul);
        }

        // Metrics Summary
        if (Array.isArray(data.metrics_summary) && data.metrics_summary.length > 0) {
            const metricsDiv = document.createElement('div');
            metricsDiv.className = 'rag-metrics';
            data.metrics_summary.forEach(m => {
                const line = document.createElement('div');
                line.textContent = (typeof m === 'object') ? JSON.stringify(m) : String(m);
                metricsDiv.appendChild(line);
            });
            container.appendChild(metricsDiv);
        }

        // Citations
        if (Array.isArray(data.citations) && data.citations.length > 0) {
            const citationsDiv = document.createElement('div');
            citationsDiv.className = 'rag-citations';
            const title = document.createElement('h4');
            title.textContent = 'Sources';
            citationsDiv.appendChild(title);
            data.citations.forEach(c => {
                const wrapper = document.createElement('div');
                wrapper.className = 'citation';
                const meta = [c.speaker, c.date, c.timestamp].filter(Boolean).join(' • ');
                wrapper.innerHTML = `<div class='meta'>${meta}</div><div class='snippet'>${c.snippet}</div>`;
                citationsDiv.appendChild(wrapper);
            });
            container.appendChild(citationsDiv);
        }

        // Follow-ups
        if (Array.isArray(data.follow_ups) && data.follow_ups.length > 0) {
            const followUpsDiv = document.createElement('div');
            followUpsDiv.className = 'follow-ups';
            data.follow_ups.forEach(f => {
                const button = document.createElement('button');
                button.textContent = f;
                button.onclick = () => handleRagQuery(f);
                followUpsDiv.appendChild(button);
            });
            container.appendChild(followUpsDiv);
        }
    }


    function loadAnalysisDetails(analysisId) {
        document.querySelectorAll('#analysesList li').forEach(li => {
            li.classList.toggle('active', li.dataset.id == analysisId);
        });

        fetch(`${API_BASE_URL}/analyses/${analysisId}`)
            .then(response => response.json())
            .then(data => {
                mainTitle.textContent = `Details for ${data.source_filename}`;
                renderProfileChart(data);
                renderResultsTable(data.utterances);
                profileChartContainer.style.display = 'block';
                resultsTable.style.display = 'table';
            })
            .catch(error => console.error(`Error loading analysis ${analysisId}:`, error));
    }

    function populateTrendsMetricSelector() {
        const metrics = Object.keys(columnMapping);
        const uniqueMetrics = [...new Set(metrics)].sort();
        trendsMetricSelector.innerHTML = '';
        uniqueMetrics.forEach(metric => {
            const option = document.createElement('option');
            option.value = metric;
            option.textContent = getDisplayName(metric);
            trendsMetricSelector.appendChild(option);
        });
    }

    function renderTrendsChart() {
        const metric = trendsMetricSelector.value;
        const period = trendsPeriodSelector.value;
        if (!metric) return;

        fetch(`${API_BASE_URL}/api/trends?metric=${metric}&period=${period}`)
            .then(response => response.json())
            .then(data => {
                if (trendsChart) trendsChart.destroy();
                trendsChart = new Chart(trendsChartCanvas, {
                    type: 'line',
                    data: {
                        labels: data.labels,
                        datasets: data.datasets.map((ds, i) => ({
                            ...ds,
                            borderColor: `hsl(${(i * 360 / data.datasets.length)}, 70%, 50%)`,
                            fill: false,
                        }))
                    },
                    options: {
                        responsive: true,
                        plugins: { title: { display: true, text: `Speaker Trends for ${getDisplayName(metric)} (${period})` } },
                        scales: { y: { beginAtZero: true } }
                    }
                });
            })
            .catch(error => console.error('Error rendering trends chart:', error));
    }

    function renderProfileChart(analysisData) {
        if (!analysisData || !analysisData.utterances || analysisData.utterances.length === 0) return;
        const numUtterances = analysisData.utterances.length;
        const aggregatedAverages = {};
        const firstUtterance = analysisData.utterances[0];
        Object.keys(firstUtterance.aggregated_scores).forEach(key => { aggregatedAverages[key] = 0; });
        analysisData.utterances.forEach(u => {
            for (const key in u.aggregated_scores) {
                if (aggregatedAverages.hasOwnProperty(key)) aggregatedAverages[key] += u.aggregated_scores[key];
            }
        });
        for (const key in aggregatedAverages) { aggregatedAverages[key] /= numUtterances; }

        if (performanceChart) performanceChart.destroy();
        performanceChart = new Chart(performanceChartCanvas, {
            type: 'radar',
            data: {
                labels: Object.keys(aggregatedAverages).map(l => getDisplayName(l)),
                datasets: [{
                    label: 'Average Score Profile',
                    data: Object.values(aggregatedAverages),
                    fill: true,
                    backgroundColor: 'rgba(79, 70, 229, 0.2)',
                    borderColor: 'rgb(79, 70, 229)',
                }]
            },
            options: { responsive: true, plugins: { title: { display: true, text: 'Overall Performance Profile' } } }
        });
    }

    function renderResultsTable(utterances) {
        resultsThead.innerHTML = '';
        resultsTbody.innerHTML = '';
        if (!utterances || utterances.length === 0) return;

        const firstUtterance = utterances[0];
        
        const predictionKeys = Object.keys(firstUtterance.predictions)
            .filter(key => !key.endsWith('.1') && !key.endsWith('_1'));
            
        const aggregatedKeys = Object.keys(firstUtterance.aggregated_scores)
            .filter(key => !key.endsWith('.1') && !key.endsWith('_1'));

        const headers = ['date', 'timestamp', 'speaker', 'text', 'sa_labels', ...predictionKeys.sort(), ...aggregatedKeys.sort()];
        
        const headerRow = document.createElement('tr');
        headers.forEach(header => {
            const th = document.createElement('th');
            if (header === 'sa_labels') {
                th.textContent = 'Situational Awareness Categories';
            } else {
                th.textContent = getDisplayName(header);
            }
            headerRow.appendChild(th);
        });
        resultsThead.appendChild(headerRow);

        utterances.forEach(utterance => {
            const tr = document.createElement('tr');
            headers.forEach(header => {
                const td = document.createElement('td');
                let value;
                if (header === 'sa_labels') {
                    value = utterance.sa_labels ? utterance.sa_labels.join(', ') : 'N/A';
                } else {
                    value = utterance[header] ?? utterance.aggregated_scores[header] ?? utterance.predictions[header] ?? 'N/A';
                }
                td.textContent = value;
                tr.appendChild(td);
            });
            resultsTbody.appendChild(tr);
        });
    }

    function updateStatus(message, color) {
        statusDiv.textContent = message;
        statusDiv.style.color = color;
    }
});