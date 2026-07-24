import React, { useState } from 'react';
import type { UpcomingElection } from '../services/api';
import { Search, Filter, Sparkles, ChevronRight, BarChart2 } from 'lucide-react';
import Plot from 'react-plotly.js';

interface ForecastShowcaseProps {
  elections: UpcomingElection[];
  onSelectElection: (id: string) => void;
}

export const ForecastShowcase: React.FC<ForecastShowcaseProps> = ({ elections, onSelectElection }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedYear, setSelectedYear] = useState<string>('All');
  const [selectedType, setSelectedType] = useState<string>('All');
  const [selectedRegion, setSelectedRegion] = useState<string>('All');

  // Featured elections (prominent showcase cards for key upcoming 2025/2026/2028 races)
  const featuredIds = ['upcoming_USA_2028_Pres', 'upcoming_FRA_2027_Pres', 'upcoming_DEU_2025_Leg', 'upcoming_GBR_2028_Leg', 'upcoming_BRA_2026_Pres', 'upcoming_CAN_2025_Leg'];
  const featured = elections.filter((e) => featuredIds.includes(e.election_id) || e.year === 2025).slice(0, 4);

  // Filter elections
  const filtered = elections.filter((e) => {
    const matchesSearch =
      e.country_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      e.country_code.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesYear = selectedYear === 'All' || e.year.toString() === selectedYear;
    const matchesType = selectedType === 'All' || e.election_type === selectedType;
    const matchesRegion = selectedRegion === 'All' || e.region === selectedRegion;
    return matchesSearch && matchesYear && matchesType && matchesRegion;
  });

  const regions = Array.from(new Set(elections.map((e) => e.region))).filter(Boolean);
  const years = Array.from(new Set(elections.map((e) => e.year.toString()))).sort();

  return (
    <div className="max-w-7xl mx-auto px-6 space-y-8 pb-12">
      {/* Featured Showcase Header */}
      <div>
        <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-brandOrange" /> High-Impact Forecast Showcase
        </h2>

        {/* Featured Card Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {featured.map((item) => {
            const probPct = (item.raw_probability * 100).toFixed(1);
            const isIncumbent = item.predicted_winner === 'Incumbent';
            return (
              <div
                key={item.election_id}
                onClick={() => onSelectElection(item.election_id)}
                className="glass-card p-5 rounded-2xl cursor-pointer flex flex-col justify-between group"
              >
                <div>
                  {/* Card Top Badges */}
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                      {item.country_code}
                    </span>
                    <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-orange-500/10 text-orange-400 border border-orange-500/20">
                      {item.year} • {item.election_type}
                    </span>
                  </div>

                  {/* Country Name */}
                  <h3 className="text-base font-bold text-white group-hover:text-brandOrange transition mb-1">
                    {item.country_name}
                  </h3>
                  <p className="text-xs text-slate-400 mb-4">{item.region}</p>

                  {/* Mini Gauge Chart */}
                  <div className="w-full h-28 my-1 flex items-center justify-center">
                    <Plot
                      data={[
                        {
                          type: 'indicator',
                          mode: 'gauge+number',
                          value: parseFloat(probPct),
                          number: { suffix: '%', font: { size: 18, color: '#F8FAFC' } },
                          gauge: {
                            axis: { range: [0, 100], visible: false },
                            bar: { color: isIncumbent ? '#10B981' : '#F26419', thickness: 0.6 },
                            bgcolor: '#1E222A',
                            bordercolor: 'transparent',
                          },
                        },
                      ]}
                      layout={{
                        autosize: true,
                        margin: { l: 20, r: 20, t: 10, b: 10 },
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                      }}
                      config={{ responsive: true, displayModeBar: false }}
                      style={{ width: '100%', height: '100%' }}
                    />
                  </div>
                </div>

                {/* Card Bottom Outcome */}
                <div className="pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs">
                  <div>
                    <span className="text-slate-400">Favored: </span>
                    <span className={`font-semibold ${isIncumbent ? 'text-emerald-400' : 'text-orange-400'}`}>
                      {item.predicted_winner}
                    </span>
                  </div>
                  <span className="text-slate-400 flex items-center gap-1 group-hover:translate-x-1 transition">
                    Drivers <ChevronRight className="w-3.5 h-3.5" />
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Main Upcoming Predictions Table Section */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800/80 space-y-4">
        {/* Table Filters */}
        <div className="flex flex-col md:flex-row items-center justify-between gap-4 pb-2">
          <div className="relative w-full md:w-80">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              placeholder="Search country or code..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-9 pr-4 py-2 text-xs rounded-xl bg-slate-900/80 border border-slate-700 text-white placeholder-slate-500 focus:outline-none focus:border-brandOrange"
            />
          </div>

          <div className="flex items-center gap-3 w-full md:w-auto flex-wrap">
            <div className="flex items-center gap-1.5 text-xs text-slate-400">
              <Filter className="w-3.5 h-3.5" /> Filters:
            </div>

            {/* Year Selector */}
            <select
              value={selectedYear}
              onChange={(e) => setSelectedYear(e.target.value)}
              className="px-3 py-1.5 text-xs rounded-lg bg-slate-900 border border-slate-700 text-slate-200 focus:outline-none focus:border-brandOrange"
            >
              <option value="All">All Years</option>
              {years.map((y) => (
                <option key={y} value={y}>
                  {y}
                </option>
              ))}
            </select>

            {/* Type Selector */}
            <select
              value={selectedType}
              onChange={(e) => setSelectedType(e.target.value)}
              className="px-3 py-1.5 text-xs rounded-lg bg-slate-900 border border-slate-700 text-slate-200 focus:outline-none focus:border-brandOrange"
            >
              <option value="All">All Types</option>
              <option value="Executive">Executive</option>
              <option value="Legislative">Legislative</option>
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

        {/* Upcoming Table */}
        <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-900/40">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900/80 text-slate-400 font-medium">
                <th className="py-3 px-4">Country</th>
                <th className="py-3 px-4">Code</th>
                <th className="py-3 px-4">Year</th>
                <th className="py-3 px-4">Type</th>
                <th className="py-3 px-4">Region</th>
                <th className="py-3 px-4 text-center">Incumbent Win Prob</th>
                <th className="py-3 px-4 text-center">Forecast Winner</th>
                <th className="py-3 px-4 text-center">Confidence</th>
                <th className="py-3 px-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-200">
              {filtered.length === 0 ? (
                <tr>
                  <td colSpan={9} className="text-center py-8 text-slate-500">
                    No upcoming elections found matching filters.
                  </td>
                </tr>
              ) : (
                filtered.map((row) => {
                  const probPct = (row.raw_probability * 100).toFixed(1);
                  const isIncumbent = row.predicted_winner === 'Incumbent';
                  return (
                    <tr
                      key={row.election_id}
                      onClick={() => onSelectElection(row.election_id)}
                      className="hover:bg-slate-800/40 transition cursor-pointer"
                    >
                      <td className="py-3 px-4 font-bold text-white">{row.country_name}</td>
                      <td className="py-3 px-4 font-mono text-slate-400">{row.country_code}</td>
                      <td className="py-3 px-4">{row.year}</td>
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
                      <td className="py-3 px-4 text-center">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                            isIncumbent
                              ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                              : 'bg-orange-500/10 text-orange-400 border border-orange-500/20'
                          }`}
                        >
                          {row.predicted_winner}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-center font-mono text-slate-300">
                        {(row.adjusted_confidence * 100).toFixed(1)}%
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
