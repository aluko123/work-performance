import type { Analysis, ColumnMapping } from '../App';

export interface AnalysisDisplayProps {
    analysis: Analysis;
    columnMapping: ColumnMapping;
}


export function AnalysisDisplay({ analysis, columnMapping }: AnalysisDisplayProps) {

    if (!analysis.utterances || analysis.utterances.length === 0) {
        return (
            <div className='analysis-section'>
                <h2>Analysis for: {analysis.source_filename}</h2>
                <p>No utterances were found in this document.</p>
            </div>
        );
    }    

    const metricKeys = Array.from(new Set(
        analysis.utterances.flatMap(utt => Object.keys(utt.predictions || {}))
    ));

    const aggregatedScoreKeys = Array.from(new Set(
        analysis.utterances.flatMap(utt => Object.keys(utt.aggregated_scores || {}))
    ));


    return (
        <div className='analysis-section'>
            <h2>Analysis for: {analysis.source_filename}</h2>
            <div className='utterance-table'>
                <table>
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Timestamp</th>
                            <th>Speaker</th>
                            <th>Text</th>
                            <th>Situational Awareness Categories</th>
                            {/* create headers dynamically */}
                            {metricKeys.map(key => (
                                <th key={key}>{(columnMapping[key] && columnMapping[key].original_name) || key}</th>
                            ))}
                            {aggregatedScoreKeys.map(key => (
                                <th key={key}>{(columnMapping[key] && columnMapping[key].original_name) || key}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {analysis.utterances.map((utt, index) => (
                            <tr key={index}>
                                <td>{utt.date || 'N/A'}</td>
                                <td>{utt.timestamp || 'N/A'}</td>
                                <td>{utt.speaker}</td>
                                <td>{utt.text}</td>
                                <td>{utt.sa_labels?.join(', ') || 'N/A'}</td>
                                {/*render other scores*/}
                                {metricKeys.map(key => (
                                    <td key={key}>{utt.predictions?.[key] ?? 'N/A'}</td>
                                ))}
                                {aggregatedScoreKeys.map(key=> (
                                    <td key={key}>{utt.aggregated_scores?.[key] ?? 'N/A'}</td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}