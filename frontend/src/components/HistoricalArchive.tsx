import React, { useState } from 'react';
import type { HistoricalElection } from '../services/api';
import { Search, Filter, History, CheckCircle2, XCircle, BarChart2 } from 'lucide-react';

interface HistoricalArchiveProps {
  elections: HistoricalElection[];
  onSelectElection: (id: string) => void;
}

export const HistoricalArchive: React.FC<HistoricalArchiveProps> = ({ elections, onSelectElection }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedRegion, setSelectedRegion] = useState('All');
  const [selectedStatus, setSelectedStatus] = useState('All');

  const filtered = elections.filter((e) => {
    const matchesSearch =
      e.country_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      e.country_code.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesRegion = selectedRegion === 'All' || e.region === selectedRegion;
    const matchesStatus =
      selectedStatus === 'All' ||
      (selectedStatus === 'Correct' && e.is_correct === 1) ||
      (selectedStatus === 'Incorrect' && e.is_correct === 0);

    return matchesSearch && matchesRegion && matchesStatus;
  });

  const regions = Array.from(new Set(elections.map((e) => e.region))).filter(Boolean);

  const totalEvaluated = elections.length;
  const correctCount = elections.filter((e) => e.is_correct === 1).length;
  const accuracyPct = ((correctCount / totalEvaluated) * 100).toFixed(2);

  return (
    <div className="max-w-7xl mx-auto px-6 space-y-6 pb-12">
      {/* Archive Header Banner */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800/80 flex flex-col md:flex-row items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2 m-0">
            <History className="w-5 h-5 text-purple-400" /> Historical Out-of-Sample Archive
          </h2>
          <p className="text-xs text-slate-400 mt-1 m-0">
            Evaluated using 5-fold cross-validated out-of-sample predictions across historical election cycles (1970–2026).
          </p>
        </div>

        <div className="flex items-center gap-4 bg-slate-900/80 px-4 py-2.5 rounded-xl border border-slate-800 text-xs">
          <div>
            <span className="text-slate-400">Evaluated: </span>
            <span className="font-bold text-white font-mono">{totalEvaluated}</span>
          </div>
          <div className="w-px h-4 bg-slate-700" />
          <div>
            <span className="text-slate-400">Correct: </span>
            <span className="font-bold text-emerald-400 font-mono">{correctCount}</span>
          </div>
          <div className="w-px h-4 bg-slate-700" />
          <div>
            <span className="text-slate-400">Accuracy: </span>
            <span className="font-bold text-brandOrange font-mono">{accuracyPct}%</span>
          </div>
        </div>
      </div>

      {/* Main Table Section */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800/80 space-y-4">
        {/* Table Filters */}
        <div className="flex flex-col md:flex-row items-center justify-between gap-4 pb-2">
          <div className="relative w-full md:w-80">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              placeholder="Search historical country or code..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-9 pr-4 py-2 text-xs rounded-xl bg-slate-900/80 border border-slate-700 text-white placeholder-slate-500 focus:outline-none focus:border-brandOrange"
            />
          </div>

          <div className="flex items-center gap-3 w-full md:w-auto flex-wrap">
            <div className="flex items-center gap-1.5 text-xs text-slate-400">
              <Filter className="w-3.5 h-3.5" /> Filters:
            </div>

            {/* Match Status Selector */}
            <select
              value={selectedStatus}
              onChange={(e) => setSelectedStatus(e.target.value)}
              className="px-3 py-1.5 text-xs rounded-lg bg-slate-900 border border-slate-700 text-slate-200 focus:outline-none focus:border-brandOrange"
            >
              <option value="All">All Results</option>
              <option value="Correct">Correct Only</option>
              <option value="Incorrect">Incorrect Only</option>
            </select>

            {/* Region Selector */}
            <select
              value={selectedRegion}
              onChange={(e) => setSelectedRegion(e.target.value)}
              className="px-3 py-1.5 text-xs rounded-lg bg-slate-900 border border-slate-700 text-slate-200 focus:outline-none focus:border-brandOrange"
            >
              <option value="All">All Regions</option>
              {regions.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Historical Table */}
        <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-900/40">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900/80 text-slate-400 font-medium">
                <th className="py-3 px-4">Country</th>
                <th className="py-3 px-4">Code</th>
                <th className="py-3 px-4">Year</th>
                <th className="py-3 px-4">Type</th>
                <th className="py-3 px-4">Region</th>
                <th className="py-3 px-4 text-center">Incumbent Prob</th>
                <th className="py-3 px-4 text-center">Forecast Winner</th>
                <th className="py-3 px-4 text-center">Actual Outcome</th>
                <th className="py-3 px-4 text-center">Prediction Match</th>
                <th className="py-3 px-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-200">
              {filtered.length === 0 ? (
                <tr>
                  <td colSpan={10} className="text-center py-8 text-slate-500">
                    No historical elections found matching filters.
                  </td>
                </tr>
              ) : (
                filtered.map((row) => {
                  const probPct = (row.raw_probability * 100).toFixed(1);
                  const isCorrect = row.is_correct === 1;
                  const actualOutcomeText = row.target_outcome_int === 1 ? 'Incumbent Victory' : 'Challenger Victory';

                  return (
                    <tr
                      key={row.election_id}
                      onClick={() => onSelectElection(row.election_id)}
                      className="hover:bg-slate-800/40 transition cursor-pointer"
                    >
                      <td className="py-3 px-4 font-bold text-white">{row.country_name}</td>
                      <td className="py-3 px-4 font-mono text-slate-400">{row.country_code}</td>
                      <td className="py-3 px-4 font-mono">{row.year}</td>
                      <td className="py-3 px-4">
                        <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-800 text-slate-300 border border-slate-700">
                          {row.election_type}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-slate-400">{row.region}</td>
                      <td className="py-3 px-4 text-center font-mono font-bold">
                        <span className={row.raw_probability >= 0.5 ? 'text-emerald-400' : 'text-orange-400'}>
                          {probPct}%
                        </span>
                      </td>
                      <td className="py-3 px-4 text-center font-medium">{row.predicted_winner}</td>
                      <td className="py-3 px-4 text-center text-slate-300">{actualOutcomeText}</td>
                      <td className="py-3 px-4 text-center">
                        {isCorrect ? (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                            <CheckCircle2 className="w-3 h-3" /> Correct
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold bg-red-500/10 text-red-400 border border-red-500/20">
                            <XCircle className="w-3 h-3" /> Incorrect
                          </span>
                        )}
                      </td>
                      <td className="py-3 px-4 text-right">
                        <button className="px-2.5 py-1 rounded bg-slate-800 hover:bg-brandOrange text-slate-300 hover:text-white transition text-[11px] font-medium inline-flex items-center gap-1">
                          <BarChart2 className="w-3 h-3" /> SHAP Drivers
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
