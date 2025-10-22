import { Link, Route, Routes, useLocation } from 'react-router-dom';
import { cn } from './lib/utils';
import { ModeToggle } from './components/mode-toggle';
import UploadView from './routes/UploadView';
import AnalysesView from './routes/AnalysesView';
import TrendsView from './routes/TrendsView';
import { BarChart3, FileText, TrendingUp, Upload } from 'lucide-react';

const navigation = [
  { name: 'Upload', href: '/', icon: Upload },
  { name: 'Analyses', href: '/analyses', icon: FileText },
  { name: 'Trends', href: '/trends', icon: TrendingUp },
];

function App() {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-background">
      <div className="flex h-screen">
        {/* Sidebar */}
        <div className="hidden md:flex md:w-64 md:flex-col">
          <div className="flex flex-col flex-grow pt-5 pb-4 overflow-y-auto bg-card border-r">
            <div className="flex items-center flex-shrink-0 px-4">
              <BarChart3 className="h-8 w-8 text-primary" />
              <span className="ml-2 text-xl font-bold">Work Analytics</span>
            </div>
            <div className="mt-5 flex-grow flex flex-col">
              <nav className="flex-1 px-2 space-y-1">
                {navigation.map((item) => {
                  const isActive = location.pathname === item.href;
                  return (
                    <Link
                      key={item.name}
                      to={item.href}
                      className={cn(
                        'group flex items-center px-2 py-2 text-sm font-medium rounded-md transition-colors',
                        isActive
                          ? 'bg-primary text-primary-foreground'
                          : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                      )}
                    >
                      <item.icon className="mr-3 h-5 w-5" />
                      {item.name}
                    </Link>
                  );
                })}
              </nav>
            </div>
          </div>
        </div>

        {/* Main content */}
        <div className="flex flex-col flex-1 overflow-hidden">
          {/* Top bar */}
          <div className="bg-card border-b px-4 py-3 flex items-center justify-between">
            <div className="md:hidden">
              <BarChart3 className="h-6 w-6 text-primary" />
            </div>
            <div className="flex items-center space-x-4">
              <ModeToggle />
            </div>
          </div>

          {/* Page content */}
          <main className="flex-1 relative overflow-y-auto focus:outline-none">
            <div className="py-6">
              <div className="max-w-7xl mx-auto px-4 sm:px-6 md:px-8">
                <Routes location={location}>
                  <Route path="/" element={<UploadView />} />
                  <Route path="/analyses" element={<AnalysesView />} />
                  <Route path="/trends" element={<TrendsView />} />
                </Routes>
              </div>
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}

export default App;
