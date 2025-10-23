import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import './index.css'
import App from './App.tsx'
import { ThemeProvider } from './components/theme-provider'
import { UploadProvider } from './contexts/UploadContext'
import { AppStateProvider } from './contexts/AppStateContext'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 10 * 60 * 1000, // 10 minutes
    },
  },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <ThemeProvider defaultTheme="dark" storageKey="work-analytics-theme">
        <AppStateProvider>
          <UploadProvider>
            <BrowserRouter>
              <App />
            </BrowserRouter>
          </UploadProvider>
        </AppStateProvider>
      </ThemeProvider>
    </QueryClientProvider>
  </StrictMode>,
)
