import { useState, useMemo, useDeferredValue } from 'react';
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
  type ColumnFiltersState,
} from '@tanstack/react-table';
import { ArrowUpDown, ArrowUp, ArrowDown, Search, Download } from 'lucide-react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { Badge } from './ui/badge';
import type { ColumnMapping } from '../lib/types';
import type { Analysis } from '../lib/types';

export interface AnalysisDisplayProps {
  analysis: Analysis;
  columnMapping: ColumnMapping;
}

type UtteranceRow = {
  id: number;
  date: string;
  timestamp: string;
  speaker: string;
  text: string;
  sa_labels: string[];
} & Record<string, any>;

export function AnalysisDisplay({ analysis, columnMapping }: AnalysisDisplayProps) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);
  const [globalFilter, setGlobalFilter] = useState('');
  const deferredGlobalFilter = useDeferredValue(globalFilter);

  if (!analysis.utterances || analysis.utterances.length === 0) {
    return (
      <div className="text-center py-8">
        <p className="text-muted-foreground">No utterances were found in this document.</p>
      </div>
    );
  }

  const metricKeys = useMemo(() => 
    Array.from(new Set(
      analysis.utterances.flatMap(utt => Object.keys(utt.predictions || {}))
    )).filter(k => !(k.endsWith('.1') || k.endsWith('_1'))),
    [analysis.utterances]
  );

  const aggregatedScoreKeys = useMemo(() => 
    Array.from(new Set(
      analysis.utterances.flatMap(utt => Object.keys(utt.aggregated_scores || {}))
    )).filter(k => !(k.endsWith('.1') || k.endsWith('_1'))),
    [analysis.utterances]
  );

  const data: UtteranceRow[] = useMemo(() =>
    analysis.utterances.map((utt, index) => ({
      id: index,
      date: utt.date || 'N/A',
      timestamp: utt.timestamp || 'N/A',
      speaker: utt.speaker,
      text: utt.text,
      sa_labels: utt.sa_labels || [],
      ...Object.fromEntries([
        ...metricKeys.map(key => [key, utt.predictions?.[key] ?? null]),
        ...aggregatedScoreKeys.map(key => [key, utt.aggregated_scores?.[key] ?? null]),
      ]),
    })), [analysis.utterances, metricKeys, aggregatedScoreKeys]
  );

  const columns: ColumnDef<UtteranceRow>[] = useMemo(() => [
    {
      accessorKey: 'date',
      header: 'Date',
      enableGlobalFilter: false,
      cell: ({ getValue }: any) => <span className="font-mono text-sm">{getValue() as string}</span>,
    },
    {
      accessorKey: 'timestamp',
      header: 'Time',
      enableGlobalFilter: false,
      cell: ({ getValue }: any) => <span className="font-mono text-sm">{getValue() as string}</span>,
    },
    {
      accessorKey: 'speaker',
      header: 'Speaker',
      cell: ({ getValue }: any) => (
        <Badge variant="outline" className="font-medium">
          {getValue() as string}
        </Badge>
      ),
    },
    {
      accessorKey: 'text',
      header: 'Text',
      cell: ({ getValue }: any) => (
        <div className="max-w-xs truncate" title={getValue() as string}>
          {getValue() as string}
        </div>
      ),
    },
    {
      accessorKey: 'sa_labels',
      header: 'Situational Awareness',
      enableGlobalFilter: false,
      cell: ({ getValue }: any) => {
        const labels = getValue() as string[];
        return (
          <div className="flex flex-wrap gap-1">
            {labels.map((label, idx) => (
              <Badge key={idx} variant="secondary" className="text-xs">
                {label}
              </Badge>
            ))}
          </div>
        );
      },
    },
    ...metricKeys.map(key => ({
      accessorKey: key,
      header: (columnMapping[key]?.original_name) || key,
      enableGlobalFilter: false,
      cell: ({ getValue }: any) => {
        const value = getValue() as number | null;
        return value !== null ? (
          <span className="font-mono text-sm">
            {typeof value === 'number' ? value.toFixed(3) : value}
          </span>
        ) : (
          <span className="text-muted-foreground">N/A</span>
        );
      },
    })),
    ...aggregatedScoreKeys.map(key => ({
      accessorKey: key,
      header: (columnMapping[key]?.original_name) || key,
      enableGlobalFilter: false,
      cell: ({ getValue }: any) => {
        const value = getValue() as number | null;
        return value !== null ? (
          <span className="font-mono text-sm font-semibold">
            {typeof value === 'number' ? value.toFixed(3) : value}
          </span>
        ) : (
          <span className="text-muted-foreground">N/A</span>
        );
      },
    })),
  ], [metricKeys, aggregatedScoreKeys, columnMapping]);

  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onGlobalFilterChange: setGlobalFilter,
    state: {
      sorting,
      columnFilters,
      globalFilter: deferredGlobalFilter,
    },
    initialState: {
      pagination: {
        pageSize: 25,
      },
    },
  });

  const exportToCSV = () => {
    const headers = columns.map(col => (col as any).header as string);
    const rows = data.map(row => columns.map(col => {
      const value = row[(col as any).accessorKey as keyof UtteranceRow];
      if (Array.isArray(value)) return value.join('; ');
      return value?.toString() || '';
    }));

    const csvContent = [headers, ...rows]
      .map(row => row.map(cell => `"${cell}"`).join(','))
      .join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${analysis.source_filename}_analysis.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Search className="h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search utterances..."
            value={globalFilter ?? ''}
            onChange={(event) => setGlobalFilter(String(event.target.value))}
            className="max-w-sm"
          />
        </div>
        <Button onClick={exportToCSV} variant="outline" size="sm">
          <Download className="mr-2 h-4 w-4" />
          Export CSV
        </Button>
      </div>

      <div className="rounded-md border">
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHead key={header.id}>
                    {header.isPlaceholder ? null : (
                      <Button
                        variant="ghost"
                        onClick={header.column.getToggleSortingHandler()}
                        className="h-auto p-0 font-semibold hover:bg-transparent"
                      >
                        {flexRender(
                          header.column.columnDef.header,
                          header.getContext()
                        )}
                        {{
                          asc: <ArrowUp className="ml-2 h-4 w-4" />,
                          desc: <ArrowDown className="ml-2 h-4 w-4" />,
                        }[header.column.getIsSorted() as string] ?? (
                          <ArrowUpDown className="ml-2 h-4 w-4" />
                        )}
                      </Button>
                    )}
                  </TableHead>
                ))}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows?.length ? (
              table.getRowModel().rows.map((row) => (
                <TableRow
                  key={row.id}
                  data-state={row.getIsSelected() && "selected"}
                >
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id}>
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell
                  colSpan={columns.length}
                  className="h-24 text-center"
                >
                  No results.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>

      <div className="flex items-center justify-between">
        <div className="text-sm text-muted-foreground">
          Showing {table.getRowModel().rows.length} of {data.length} utterances
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
          >
            Previous
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
          >
            Next
          </Button>
        </div>
      </div>
    </div>
  );
}
