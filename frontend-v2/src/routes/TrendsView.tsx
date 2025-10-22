import { useEffect, useRef, useState } from 'react';
import { Chart } from 'chart.js/auto';
import { getTrends } from '../lib/api';
import type { ColumnMapping } from '../lib/types';

export default function TrendsView() {
  const [columnMapping, setColumnMapping] = useState<ColumnMapping>({});
  const [metric, setMetric] = useState<string>('');
  const [period, setPeriod] = useState<'daily' | 'weekly'>('daily');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const chartRef = useRef<Chart | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const resp = await fetch('/column_name_mapping.json');
        const map: ColumnMapping = await resp.json();
        setColumnMapping(map);
        const first = Object.keys(map)[0];
        if (first) setMetric(first);
      } catch (e) {
        setError('Failed to load column mapping');
      }
    })();
  }, []);

  useEffect(() => {
    if (!metric) return;
    let mounted = true;
    (async () => {
      setLoading(true); setError(null);
      try {
        const data = await getTrends(metric, period);
        if (!mounted) return;
        if (chartRef.current) chartRef.current.destroy();
        if (!canvasRef.current) return;
        chartRef.current = new Chart(canvasRef.current, {
          type: 'line',
          data: {
            labels: data.labels,
            datasets: data.datasets.map((ds, i) => ({
              ...ds,
              borderColor: `hsl(${i * 47},70%,50%)`,
              fill: false,
            }))
          },
          options: { responsive: true, plugins: { title: { display: true, text: `Speaker Trends (${getDisplayName(metric, columnMapping)} - ${period})` } }, scales: { y: { beginAtZero: true } } }
        });
      } catch (e: any) {
        if (mounted) setError(e.message);
      } finally {
        if (mounted) setLoading(false);
      }
    })();
    return () => { mounted = false };
  }, [metric, period]);

  return (
    <div className="app-container">
      <h2>Speaker Trends</h2>
      <div className="chart-controls" style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
        <label>Metric:</label>
        <select value={metric} onChange={(e) => setMetric(e.target.value)}>
          {Object.keys(columnMapping).map(key => (
            <option key={key} value={key}>{getDisplayName(key, columnMapping)}</option>
          ))}
        </select>
        <label>Period:</label>
        <select value={period} onChange={(e) => setPeriod(e.target.value as 'daily' | 'weekly')}>
          <option value="daily">Daily</option>
          <option value="weekly">Weekly</option>
        </select>
      </div>
      {loading && <div className='loading-indicator'>Loading chart...</div>}
      {error && <div className='error-message'>{error}</div>}
      <div className="chart-container" style={{ display: 'block' }}>
        <canvas ref={canvasRef} />
      </div>
    </div>
  );
}

function getDisplayName(raw: string, mapping: ColumnMapping): string {
  return (mapping?.[raw]?.original_name) || raw;
}

