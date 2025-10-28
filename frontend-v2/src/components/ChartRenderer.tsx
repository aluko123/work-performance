import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';

interface ChartDataset {
  name: string;
  data: number[];
}

interface ChartConfig {
  xAxisLabel: string;
  yAxisLabel: string;
  title: string;
  colors?: string[];
  yDomain?: [number, number];
}

interface ChartData {
  labels: string[];
  datasets: ChartDataset[];
}

interface Chart {
  type: 'line' | 'bar' | 'grouped_bar';
  data: ChartData;
  config: ChartConfig;
}

interface ChartRendererProps {
  chart: Chart;
}

export function ChartRenderer({ chart }: ChartRendererProps) {
  const { type, data, config } = chart;

  // Transform data from backend format to Recharts format
  const chartData = data.labels.map((label, idx) => {
    const point: Record<string, string | number> = { name: label };
    data.datasets.forEach((dataset) => {
      point[dataset.name] = dataset.data[idx];
    });
    return point;
  });

  const colors = config.colors || [
    '#8884d8',
    '#82ca9d',
    '#ffc658',
    '#ff7c7c',
    '#8dd1e1',
  ];

  if (type === 'line') {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="text-base font-medium">{config.title}</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis
                dataKey="name"
                label={{
                  value: config.xAxisLabel,
                  position: 'insideBottom',
                  offset: -5,
                }}
                className="text-xs"
              />
              <YAxis
                domain={config.yDomain || [0, 'auto']}
                label={{
                  value: config.yAxisLabel,
                  angle: -90,
                  position: 'insideLeft',
                }}
                className="text-xs"
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--card))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '6px',
                }}
              />
              <Legend />
              {data.datasets.map((dataset, idx) => (
                <Line
                  key={dataset.name}
                  type="monotone"
                  dataKey={dataset.name}
                  stroke={colors[idx % colors.length]}
                  strokeWidth={2}
                  dot={{ r: 4 }}
                  activeDot={{ r: 6 }}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    );
  }

  if (type === 'bar' || type === 'grouped_bar') {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="text-base font-medium">{config.title}</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis
                dataKey="name"
                label={{
                  value: config.xAxisLabel,
                  position: 'insideBottom',
                  offset: -5,
                }}
                className="text-xs"
              />
              <YAxis
                domain={config.yDomain || [0, 'auto']}
                label={{
                  value: config.yAxisLabel,
                  angle: -90,
                  position: 'insideLeft',
                }}
                className="text-xs"
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--card))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '6px',
                }}
              />
              <Legend />
              {data.datasets.map((dataset, idx) => (
                <Bar
                  key={dataset.name}
                  dataKey={dataset.name}
                  fill={colors[idx % colors.length]}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    );
  }

  return null;
}
