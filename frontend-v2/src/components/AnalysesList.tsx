import { useInfiniteQuery } from '@tanstack/react-query';
import { useEffect } from 'react';
import { useInView } from 'react-intersection-observer';
import { getAnalyses } from '../lib/api';
import { Button } from './ui/button';
import { Loader2, AlertCircle } from 'lucide-react';
import type { AnalysesResponse } from '../lib/types';

interface AnalysesListProps {
  selectedId: number | null;
  onSelect: (id: number) => void;
}

export function AnalysesList({ selectedId, onSelect }: AnalysesListProps) {
  const { ref, inView } = useInView();

  const {
    data,
    error,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading,
    refetch,
  } = useInfiniteQuery({
    queryKey: ['analyses'] as const,
    queryFn: ({ pageParam }: { pageParam: unknown }) => getAnalyses(pageParam as number, 20),
    initialPageParam: 1,
    getNextPageParam: (lastPage: AnalysesResponse) => {
      return lastPage.has_more ? lastPage.page + 1 : undefined;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
  });



  // Load more when scrolling to bottom
  useEffect(() => {
    if (inView && hasNextPage && !isFetchingNextPage) {
      fetchNextPage();
    }
  }, [inView, hasNextPage, isFetchingNextPage, fetchNextPage]);

  const allItems = data?.pages.flatMap((page: AnalysesResponse) => page.items) || [];

  if (error) {
    return (
      <div className="analyses-list p-4">
        <h2 className="text-lg font-semibold mb-4">Past Analyses</h2>
        <div className="flex items-center gap-2 text-destructive">
          <AlertCircle className="h-4 w-4" />
          <span>Failed to load analyses</span>
        </div>
        <Button variant="outline" size="sm" onClick={() => refetch()} className="mt-2">
          Try Again
        </Button>
      </div>
    );
  }

  return (
    <div className="analyses-list p-4">
      <h2 className="text-lg font-semibold mb-4">Past Analyses</h2>

      {isLoading && allItems.length === 0 && (
        <div className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 animate-spin" />
          <span className="ml-2">Loading analyses...</span>
        </div>
      )}

      <div className="space-y-2">
        {allItems.map(a => (
          <div
            key={a.id}
            onClick={() => onSelect(a.id)}
            className={`p-3 rounded-lg border cursor-pointer transition-colors ${
              selectedId === a.id
                ? 'bg-primary text-primary-foreground border-primary'
                : 'bg-card hover:bg-accent border-border'
            }`}
          >
            <div className="font-medium truncate">{a.source_filename}</div>
            {a.created_at && (
              <div className="text-sm text-muted-foreground">
                {new Date(a.created_at).toLocaleDateString()} at{' '}
                {new Date(a.created_at).toLocaleTimeString()}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Infinite scroll trigger */}
      <div ref={ref} className="py-4">
        {isFetchingNextPage && (
          <div className="flex items-center justify-center">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span className="ml-2 text-sm text-muted-foreground">Loading more...</span>
          </div>
        )}
        {!hasNextPage && allItems.length > 0 && (
          <div className="text-center text-sm text-muted-foreground py-2">
            No more analyses to load
          </div>
        )}
      </div>
    </div>
  );
}

