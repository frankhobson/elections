import React, { useState } from 'react';
import type { CountryAccuracy } from '../services/api';
import { Globe, Search, Filter, ShieldAlert } from 'lucide-react';
import Plot from 'react-plotly.js';

interface CountryAccuracyViewProps {
  countries: CountryAccuracy[];
}

export const CountryAccuracyView: React.FC<CountryAccuracyViewProps> = ({ countries }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedRegion, setSelectedRegion] = useState('All');

  const filtered = countries.filter((c) => {
    const matchesSearch =
      c.country_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      c.country_code.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesRegion = selectedRegion === 'All' || c.region === selectedRegion;
    return matchesSearch && matchesRegion;
  });

  const regions = Array.from(new Set(countries.map((c) => c.region))).filter(Boolean);

  // Choropleth map data (filter out countries with < 3 elections to avoid distortion)
  const mapCountries = countries.filter((c) => c.total >= 3);
  const countryCodes = mapCountries.map((c) => c.country_code);
  const accuracies = mapCountries.map((c) => c.accuracy_pct);
  const hoverTexts = mapCountries.map(
    (c) =>
      `<b>${c.country_name} (${c.country_code})</b><br>` +
      `Region: ${c.region}<br>` +
      `Overall Accuracy: ${c.accuracy_pct}% (${c.correct}/${c.total})<br>` +
      `Executive: ${c.exec_correct}/${c.exec_total}<br>` +
      `Legislative: ${c.leg_correct}/${c.leg_total}`
  );

  return (
    <div className="max-w-7xl mx-auto px-6 space-y-6 pb-12">
      {/* Header Banner */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800/80 flex flex-col md:flex-row items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2 m-0">
            <Globe className="w-5 h-5 text-brandCyan" /> Out-of-Sample Accuracy by Country
          </h2>
          <p className="text-xs text-slate-400 mt-1 m-0">
            Cross-validated predictive accuracy aggregated across 168 historical national election portfolios.
          </p>
        </div>

        <div className="flex items-center gap-2 bg-slate-900/80 px-3.5 py-2 rounded-xl border border-slate-800 text-xs text-slate-400">
          <ShieldAlert className="w-4 h-4 text-amber-400" />
          <span>Countries with &lt;3 elections grayed out on map to prevent sample distortion.</span>
        </div>
      </div>

      {/* Accuracy Map */}
      <div className="glass-panel p-4 rounded-2xl border border-slate-800/80">
        <div className="w-full h-[480px]">
          <Plot
            data={[
              {
                type: 'choropleth',
                locationmode: 'ISO-3',
                locations: countryCodes,
                z: accuracies,
                text: hoverTexts,
                hoverinfo: 'text',
                colorscale: [
                  [0.0, '#DC2626'], // Low accuracy (Red)
                  [0.5, '#F59E0B'], // Mid accuracy (Amber)
                  [0.75, '#10B981'], // High accuracy (Emerald)
                  [1.0, '#06B6D4'], // Perfect accuracy (Cyan)
                ],
                zmin: 0.0,
                zmax: 100.0,
                colorbar: {
                  title: { text: 'Accuracy (%)', font: { color: '#F8FAFC', size: 11 } },
                  tickfont: { color: '#94A3B8', size: 10 },
                  len: 0.8,
                  x: 1.02,
                },
                marker: {
                  line: {
                    color: '#1E222A',
                    width: 0.8,
                  },
                },
              },
            ]}
            layout={{
              autosize: true,
              margin: { l: 0, r: 0, t: 0, b: 0 },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              geo: {
                showframe: false,
                showcoastlines: true,
                coastlinecolor: '#2D3748',
                showland: true,
                landcolor: '#1E222A',
                showocean: true,
                oceancolor: '#0E1117',
                showlakes: true,
                lakecolor: '#0E1117',
                bgcolor: 'rgba(0,0,0,0)',
                projection: {
                  type: 'natural earth',
                },
              },
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%', height: '100%' }}
          />
        </div>
      </div>

      {/* Ranked Country Table Section */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800/80 space-y-4">
        {/* Table Filters */}
        <div className="flex flex-col md:flex-row items-center justify-between gap-4 pb-2">
          <div className="relative w-full md:w-80">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              placeholder="Search country name or code..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-9 pr-4 py-2 text-xs rounded-xl bg-slate-900/80 border border-slate-700 text-white placeholder-slate-500 focus:outline-none focus:border-brandOrange"
            />
          </div>

          <div className="flex items-center gap-3 w-full md:w-auto">
            <div className="flex items-center gap-1.5 text-xs text-slate-400">
              <Filter className="w-3.5 h-3.5" /> Region Filter:
            </div>
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

        {/* Country Ranked Table */}
        <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-900/40">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900/80 text-slate-400 font-medium">
                <th className="py-3 px-4 text-center">Rank</th>
                <th className="py-3 px-4">Country Name</th>
                <th className="py-3 px-4">Code</th>
                <th className="py-3 px-4">Region</th>
                <th className="py-3 px-4 text-center">Total Elections</th>
                <th className="py-3 px-4 text-center">Correct Predictions</th>
                <th className="py-3 px-4 text-center">Executive Success</th>
                <th className="py-3 px-4 text-center">Legislative Success</th>
                <th className="py-3 px-4 text-right">Accuracy Rate</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-200">
              {filtered.length === 0 ? (
                <tr>
                  <td colSpan={9} className="text-center py-8 text-slate-500">
                    No country accuracy records found.
                  </td>
                </tr>
              ) : (
                filtered.map((row) => {
                  return (
                    <tr key={row.country_code} className="hover:bg-slate-800/40 transition">
                      <td className="py-3 px-4 text-center font-mono text-slate-400">#{row.rank}</td>
                      <td className="py-3 px-4 font-bold text-white">{row.country_name}</td>
                      <td className="py-3 px-4 font-mono text-slate-400">{row.country_code}</td>
                      <td className="py-3 px-4 text-slate-400">{row.region}</td>
                      <td className="py-3 px-4 text-center font-mono">{row.total}</td>
                      <td className="py-3 px-4 text-center font-mono font-bold text-emerald-400">{row.correct}</td>
                      <td className="py-3 px-4 text-center font-mono text-slate-400">
                        {row.exec_correct}/{row.exec_total}
                      </td>
                      <td className="py-3 px-4 text-center font-mono text-slate-400">
                        {row.leg_correct}/{row.leg_total}
                      </td>
                      <td className="py-3 px-4 text-right font-mono font-extrabold text-sm">
                        <span
                          className={
                            row.accuracy_pct >= 80
                              ? 'text-cyan-400'
                              : row.accuracy_pct >= 60
                              ? 'text-emerald-400'
                              : 'text-amber-400'
                          }
                        >
                          {row.accuracy_pct.toFixed(2)}%
                        </span>
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
