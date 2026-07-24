import { useEffect, useState } from 'react';
import { Header } from './components/Header';
import { HeroSection } from './components/HeroSection';
import { ShowcaseSection } from './components/ShowcaseSection';
import { MapSection } from './components/MapSection';
import { ExplorerSection } from './components/ExplorerSection';
import { MethodologySection } from './components/MethodologySection';
import { DiagnosticsSection } from './components/DiagnosticsSection';
import type {
  StatsResponse,
  UpcomingElection,
  HistoricalElection,
  CountryAccuracy,
  DiagnosticsResponse,
} from './services/api';
import { fetchStats, fetchUpcoming, fetchHistorical, fetchCountryAccuracy, fetchDiagnostics } from './services/api';
import { Loader2, AlertCircle } from 'lucide-react';

export function App() {
  const [activeSection, setActiveSection] = useState<string>('summary');
  const [selectedElectionId, setSelectedElectionId] = useState<string | null>(null);

  // API State
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [upcoming, setUpcoming] = useState<UpcomingElection[]>([]);
  const [historical, setHistorical] = useState<HistoricalElection[]>([]);
  const [countryAccuracy, setCountryAccuracy] = useState<CountryAccuracy[]>([]);
  const [diagnostics, setDiagnostics] = useState<DiagnosticsResponse | null>(null);

  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadAllData() {
      setLoading(true);
      setError(null);
      try {
        const [statsRes, upcomingRes, historicalRes, countryRes, diagRes] = await Promise.all([
          fetchStats(),
          fetchUpcoming(),
          fetchHistorical(),
          fetchCountryAccuracy(),
          fetchDiagnostics(),
        ]);

        setStats(statsRes);
        setUpcoming(upcomingRes);
        setHistorical(historicalRes);
        setCountryAccuracy(countryRes);
        setDiagnostics(diagRes);
        setLoading(false);
      } catch (err) {
        console.error('Failed to connect to FastAPI server:', err);
        setError('Failed to connect to backend server. Is FastAPI running on http://127.0.0.1:8000?');
        setLoading(false);
      }
    }

    loadAllData();
  }, []);

  const scrollToSection = (id: string) => {
    setActiveSection(id);
    const element = document.getElementById(id);
    if (element) {
      const yOffset = -80;
      const y = element.getBoundingClientRect().top + window.pageYOffset + yOffset;
      window.scrollTo({ top: y, behavior: 'smooth' });
    }
  };

  return (
    <div className="min-h-screen bg-[#F8FAFC] text-[#1E293B] flex flex-col font-sans selection:bg-blue-200">
      {/* Sticky Header Navigation matching app.py */}
      <Header activeSection={activeSection} scrollToSection={scrollToSection} />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-6 pt-6 pb-12">
        {/* Global Loading Spinner */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-24 gap-3 text-slate-500">
            <Loader2 className="w-10 h-10 animate-spin text-blue-600" />
            <p className="text-sm font-medium">Booting Global Forecast Engine...</p>
          </div>
        )}

        {/* Global Error Banner */}
        {error && !loading && (
          <div className="my-8 p-4 rounded-xl bg-red-50 border border-red-200 text-red-700 flex items-center gap-3 text-xs">
            <AlertCircle className="w-5 h-5 text-red-600 shrink-0" />
            <div>
              <p className="font-semibold">{error}</p>
              <p className="text-slate-600 mt-0.5">
                Ensure <code className="bg-red-100 px-1 py-0.5 rounded text-red-800 font-mono">uvicorn backend.main:app --port 8000</code> is running.
              </p>
            </div>
          </div>
        )}

        {!loading && !error && (
          <>
            {/* Section 1: Executive Summary & Hero & Showcase */}
            <HeroSection stats={stats} />

            <ShowcaseSection
              elections={upcoming}
              onSelectElection={(id) => setSelectedElectionId(id)}
            />

            {/* Section 2: Global Map */}
            <MapSection
              elections={upcoming}
              onSelectElection={(id) => setSelectedElectionId(id)}
            />

            {/* Section 3: Explorer & Feature Attribution Breakdown */}
            <ExplorerSection
              upcoming={upcoming}
              historical={historical}
              selectedElectionId={selectedElectionId}
              onSelectElection={(id) => setSelectedElectionId(id)}
            />

            {/* Section 4: Methodology */}
            <MethodologySection />

            {/* Section 5: Performance Diagnostics */}
            <DiagnosticsSection
              diagnostics={diagnostics}
              countryAccuracy={countryAccuracy}
            />
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 py-6 text-center text-xs text-slate-500 bg-white mt-auto">
        <p>Global Election Forecaster © 2026 • Recreated from Streamlit with FastAPI + Vite React</p>
      </footer>
    </div>
  );
}

export default App;
