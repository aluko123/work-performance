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
    const ragResultContainer = document.getElementById('ragResultContainer');
    const ragResultText = document.getElementById('ragResultText');


    // --- Constants ---
    const API_BASE_URL = 'http://localhost:8000';
    let columnMapping = {};

    // --- Initialization ---
    loadColumnMapping().then(() => {
        loadAnalyses();
        populateTrendsMetricSelector();
        renderTrendsChart();
    });

    // --- Event Listeners ---
    analyzeButton.addEventListener('click', handleAnalyzeClick);
    trendsMetricSelector.addEventListener('change', renderTrendsChart);
    trendsPeriodSelector.addEventListener('change', renderTrendsChart);
    ragQueryButton.addEventListener('click', handleRagQuery);

    // --- Core Functions ---

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

    function loadAnalyses() {
        fetch(`${API_BASE_URL}/analyses/`)
            .then(response => response.json())
            .then(data => {
                analysesList.innerHTML = '';
                data.forEach(analysis => {
                    const li = document.createElement('li');
                    li.textContent = `${analysis.source_filename} - ${new Date(analysis.created_at).toLocaleString()}`;
                    li.dataset.id = analysis.id;
                    li.addEventListener('click', () => loadAnalysisDetails(analysis.id));
                    analysesList.appendChild(li);
                });
                // Automatically load the first analysis if it exists
                if (data.length > 0) {
                    loadAnalysisDetails(data[0].id);
                }
            })
            .catch(error => console.error('Error loading analyses:', error));
    }

    function handleAnalyzeClick() {
        const file = fileInput.files[0];
        if (!file) {
            updateStatus('Please select a file first.', 'orange');
            return;
        }
        updateStatus('Analyzing... please wait.', 'blue');
        const formData = new FormData();
        formData.append('text_file', file);

        fetch(`${API_BASE_URL}/analyze_text/`, { method: 'POST', body: formData })
            .then(response => {
                if (!response.ok) return response.json().then(err => { throw new Error(err.detail || 'Analysis failed') });
                return response.json();
            })
            .then(newAnalysis => {
                updateStatus('Analysis complete.', 'green');
                loadAnalyses();
                loadAnalysisDetails(newAnalysis.id);
                renderTrendsChart(); // Refresh trends chart
            })
            .catch(error => {
                console.error('Error during analysis:', error);
                updateStatus(`Error: ${error.message}`, 'red');
            });
    }


    //handle RAG query
    function handleRagQuery() {
        const question = ragQueryInput.value;
        if (!question.trim()) {
            updateStatus('Please enter a question first.', 'orange');
            return;
        }

        ragResultText.textContent = 'Thinking...';
        ragResultContainer.style.display = 'block';

        fetch(`${API_BASE_URL}/get_insights/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question }),
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.detail || "Failed to get insights")});
            }
            return response.json();
        })
        .then(data => {
            ragResultText.textContent = data.answer;
        })
        .catch(error => {
            console.error('Error querying RAG:', error);
            ragResultText.textContent = 'An error occured while fetching insights: ' + error.message;
        });
        
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
        const headers = ['date', 'timestamp', 'speaker', 'text', ...Object.keys(firstUtterance.predictions).sort(), ...Object.keys(firstUtterance.aggregated_scores).sort()];
        const headerRow = document.createElement('tr');
        headers.forEach(header => {
            const th = document.createElement('th');
            th.textContent = getDisplayName(header);
            headerRow.appendChild(th);
        });
        resultsThead.appendChild(headerRow);

        utterances.forEach(utterance => {
            const tr = document.createElement('tr');
            headers.forEach(header => {
                const td = document.createElement('td');
                let value = utterance[header] ?? utterance.aggregated_scores[header] ?? utterance.predictions[header] ?? 'N/A';
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
