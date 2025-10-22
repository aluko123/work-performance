import { useEffect, useRef } from 'react';
import { Chart } from 'chart.js/auto';
import type { Analysis } from '../lib/types';

interface ProfileRadarProps {
  analysis: Analysis;
  label?: string;
}

export function ProfileRadar({ analysis, label = 'Average Score Profile' }: ProfileRadarProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const chartRef = useRef<Chart | null>(null);

  useEffect(() => {
    if (!analysis?.utterances?.length) return;

    const num = analysis.utterances.length;
    const keys = new Set<string>();
    for (const u of analysis.utterances) {
      Object.keys(u.aggregated_scores || {}).forEach(k => {
        if (!(k.endsWith('.1') || k.endsWith('_1'))) keys.add(k);
      });
    }
    const labels = Array.from(keys);
    const sums: Record<string, number> = Object.fromEntries(labels.map(k => [k, 0]));
    for (const u of analysis.utterances) {
      for (const k of labels) {
        sums[k] += (u.aggregated_scores?.[k] ?? 0);
      }
    }
    const data = labels.map(k => sums[k] / num);

    if (chartRef.current) chartRef.current.destroy();
    if (!canvasRef.current) return;

    chartRef.current = new Chart(canvasRef.current, {
      type: 'radar',
      data: {
        labels,
        datasets: [{
          label,
          data,
          fill: true,
          backgroundColor: 'rgba(79, 70, 229, 0.2)',
          borderColor: 'rgb(79, 70, 229)'
        }]
      },
      options: { responsive: true, plugins: { title: { display: true, text: 'Overall Performance Profile' } } }
    });

    return () => { if (chartRef.current) chartRef.current.destroy(); };
  }, [analysis]);

  return (
    <div className="chart-container" style={{ background: '#111827', borderRadius: 8, padding: '1rem', marginBottom: '1rem', border: '1px solid #333' }}>
      <canvas ref={canvasRef} />
    </div>
  );
}

