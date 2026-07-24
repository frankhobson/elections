import React, { useState, useEffect } from 'react';
import type { UpcomingElection, HistoricalElection, ShapResponse } from '../services/api';
import { fetchShapExplanation } from '../services/api';
import Plot from 'react-plotly.js';

interface ExplorerSectionProps {
  upcoming: UpcomingElection[];
  historical: HistoricalElection[];
  selectedElectionId: string | null;
  onSelectElection: (id: string) => void;
}

export const ExplorerSection: React.FC<ExplorerSectionProps> = ({
  upcoming,
  historical,
  selectedElectionId,
  onSelectElection,
}) => {
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [regionFilter, setRegionFilter] = useState<string>('All');
  const [typeFilter, setTypeFilter] = useState<string>('All');
  const [activeTab, setActiveTab] = useState<'upcoming' | 'historical'>('upcoming');

  // SHAP detail state
  const [shapData, setShapData] = useState<ShapResponse | null>(null);
  const [shapLoading, setShapLoading] = useState<boolean>(false);

  useEffect(() => {
    if (!selectedElectionId) return;

    setShapLoading(true);
    fetchShapExplanation(selectedElectionId)
      .then((data) => {
        setShapData(data);
        setShapLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setShapLoading(false);
      });
  }, [selectedElectionId]);

  // Filtering function
  const filterList = <T extends UpcomingElection | HistoricalElection>(list: T[]): T[] => {
    return list.filter((item) => {
      const matchesSearch =
        !searchQuery ||
        item.country_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.country_code.toLowerCase().includes(searchQuery.toLowerCase());

      const matchesType = typeFilter === 'All' || item.election_type === typeFilter;

      let matchesRegion = true;
      if (regionFilter !== 'All') {
        const regionDict: Record<string, string[]> = {
          Europe: ['Western Europe', 'Eastern Europe', 'Europe'],
          Americas: ['Latin America', 'North America', 'Caribbean', 'Americas'],
          Asia: ['Asia', 'East Asia', 'South Asia', 'Southeast Asia', 'Pacific'],
          Africa: ['Africa', 'Sub-Saharan Africa', 'North Africa'],
          'Middle East': ['Middle East', 'Middle East & North Africa', 'Western Asia'],
        };
        const validList = regionDict[regionFilter] || [];
        matchesRegion = validList.includes(item.region);
      }

      return matchesSearch && matchesType && matchesRegion;
    });
  };

  const filteredUpcoming = filterList(upcoming);
  const filteredHistorical = filterList(historical);

  return (
    <div id="database-section" className="section-container space-y-6">
      <h3 className="section-header">🔍 Interactive Election Explorer</h3>
      <div className="section-subheader">
        Filter and select any election to load its feature drivers (SHAP values) and theoretical explanation.
      </div>

      {/* Filters Bar */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1">
            🔍 Search country name...
          </label>
          <input
            type="text"
            placeholder="Search country..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full px-3.5 py-2 text-xs rounded-xl bg-slate-50 border border-slate-300 text-slate-900 focus:outline-none focus:border-blue-500 shadow-sm"
          />
        </div>

        <div>
          <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1">
            Region Filter
          </label>
          <select
            value={regionFilter}
            onChange={(e) => setRegionFilter(e.target.value)}
            className="w-full px-3.5 py-2 text-xs rounded-xl bg-slate-50 border border-slate-300 text-slate-900 font-medium focus:outline-none focus:border-blue-500 shadow-sm"
          >
            <option value="All">All</option>
            <option value="Europe">Europe</option>
            <option value="Americas">Americas</option>
            <option value="Asia">Asia</option>
            <option value="Africa">Africa</option>
            <option value="Middle East">Middle East</option>
          </select>
        </div>

        <div>
          <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1">
            Election Type
          </label>
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
            className="w-full px-3.5 py-2 text-xs rounded-xl bg-slate-50 border border-slate-300 text-slate-900 font-medium focus:outline-none focus:border-blue-500 shadow-sm"
          >
            <option value="All">All</option>
            <option value="Executive">Executive</option>
            <option value="Legislative">Legislative</option>
          </select>
        </div>
      </div>

      {/* Streamlit Tabs Navigation */}
      <div className="border-b border-slate-200 flex items-center gap-6">
        <button
          onClick={() => setActiveTab('upcoming')}
          className={`py-2.5 font-bold text-sm border-b-2 transition ${
            activeTab === 'upcoming'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          🔮 Upcoming Forecasts (Live)
        </button>
        <button
          onClick={() => setActiveTab('historical')}
          className={`py-2.5 font-bold text-sm border-b-2 transition ${
            activeTab === 'historical'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          📚 Historical Archive (Out-of-Sample CV)
        </button>
      </div>

      {/* Tab 1: Upcoming Table (Scrollable max-h-[380px] ~10.5 lines) */}
      {activeTab === 'upcoming' && (
        <div className="space-y-3">
          <p className="text-xs text-slate-600">
            Displaying <b>{filteredUpcoming.length}</b> upcoming elections scheduled from June 2026 onwards:
          </p>

          <div className="max-h-[380px] overflow-y-auto rounded-xl border border-slate-200 bg-white shadow-sm">
            <table className="w-full text-left border-collapse text-xs">
              <thead className="sticky top-0 z-10 bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold shadow-xs">
                <tr>
                  <th className="py-2.5 px-4">Country</th>
                  <th className="py-2.5 px-4">Year</th>
                  <th className="py-2.5 px-4">Type</th>
                  <th className="py-2.5 px-4">Predicted Winner</th>
                  <th className="py-2.5 px-4 text-center">Incumbent Win Prob</th>
                  <th className="py-2.5 px-4 text-center">Certainty</th>
                  <th className="py-2.5 px-4 text-center">Completeness</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {filteredUpcoming.map((row) => {
                  const isSelected = row.election_id === selectedElectionId;
                  return (
                    <tr
                      key={row.election_id}
                      onClick={() => onSelectElection(row.election_id)}
                      className={`cursor-pointer transition ${
                        isSelected ? 'bg-blue-50 border-l-4 border-l-blue-600' : 'hover:bg-slate-50'
                      }`}
                    >
                      <td className="py-2.5 px-4 font-bold text-slate-900">{row.country_name}</td>
                      <td className="py-2.5 px-4 text-slate-600">{row.year}</td>
                      <td className="py-2.5 px-4 text-slate-600">{row.election_type}</td>
                      <td className="py-2.5 px-4 font-semibold text-slate-800">{row.predicted_winner}</td>
                      <td className="py-2.5 px-4 text-center font-mono font-bold text-blue-600">
                        {(row.raw_probability * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 px-4 text-center font-mono text-slate-600">
                        {(row.adjusted_confidence * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 px-4 text-center font-mono text-slate-600">
                        {(row.data_completeness * 100).toFixed(1)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab 2: Historical Archive Table (Scrollable max-h-[380px] ~10.5 lines) */}
      {activeTab === 'historical' && (
        <div className="space-y-3">
          <p className="text-xs text-slate-600">
            Displaying <b>{filteredHistorical.length}</b> past elections. Outcomes shown are 5-fold cross-validated to simulate out-of-sample prediction:
          </p>

          <div className="max-h-[380px] overflow-y-auto rounded-xl border border-slate-200 bg-white shadow-sm">
            <table className="w-full text-left border-collapse text-xs">
              <thead className="sticky top-0 z-10 bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold shadow-xs">
                <tr>
                  <th className="py-2.5 px-4">Country</th>
                  <th className="py-2.5 px-4">Year</th>
                  <th className="py-2.5 px-4">Type</th>
                  <th className="py-2.5 px-4">Actual Outcome</th>
                  <th className="py-2.5 px-4">Model Prediction</th>
                  <th className="py-2.5 px-4 text-center">Status</th>
                  <th className="py-2.5 px-4 text-center">Incumbent Prob</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {filteredHistorical.map((row) => {
                  const isSelected = row.election_id === selectedElectionId;
                  const actualOutcomeText = row.target_outcome_int === 1 ? 'Incumbent Won' : 'Challenger Won';
                  const isCorrect = row.is_correct === 1;

                  return (
                    <tr
                      key={row.election_id}
                      onClick={() => onSelectElection(row.election_id)}
                      className={`cursor-pointer transition ${
                        isSelected ? 'bg-blue-50 border-l-4 border-l-blue-600' : 'hover:bg-slate-50'
                      }`}
                    >
                      <td className="py-2.5 px-4 font-bold text-slate-900">{row.country_name}</td>
                      <td className="py-2.5 px-4 text-slate-600">{row.year}</td>
                      <td className="py-2.5 px-4 text-slate-600">{row.election_type}</td>
                      <td className="py-2.5 px-4 text-slate-800">{actualOutcomeText}</td>
                      <td className="py-2.5 px-4 font-semibold text-slate-800">{row.predicted_winner}</td>
                      <td className="py-2.5 px-4 text-center">
                        {isCorrect ? (
                          <span className="inline-block px-2 py-0.5 rounded text-[11px] font-semibold bg-emerald-100 text-emerald-700">
                            ✅ Correct
                          </span>
                        ) : (
                          <span className="inline-block px-2 py-0.5 rounded text-[11px] font-semibold bg-red-100 text-red-700">
                            ❌ Incorrect
                          </span>
                        )}
                      </td>
                      <td className="py-2.5 px-4 text-center font-mono font-bold text-blue-600">
                        {(row.raw_probability * 100).toFixed(1)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Feature Attribution Breakdown Section (Appears when row is selected) */}
      {selectedElectionId ? (
        <div className="pt-6 border-t border-slate-200 space-y-6">
          <h3 className="section-header">
            🔎 Feature Attribution Breakdown:{' '}
            {shapData ? `${shapData.country_code} ${shapData.year} (${shapData.election_type})` : 'Loading...'}
          </h3>

          {shapLoading ? (
            <div className="py-8 text-center text-xs text-slate-500">
              Calculating SHAP values and theoretical breakdown...
            </div>
          ) : shapData ? (
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* Horizontal Bar Chart (7 cols) */}
                <div className="lg:col-span-7 bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                  <div className="w-full h-80">
                    <Plot
                      data={[
                        {
                          type: 'bar',
                          orientation: 'h',
                          x: shapData.top_features.map((f) => f.contribution),
                          y: shapData.top_features.map((f) => f.human_name),
                          marker: {
                            color: shapData.top_features.map((f) =>
                              f.direction === 'Favors Incumbent' ? '#33658A' : '#F26419'
                            ),
                          },
                        },
                      ]}
                      layout={{
                        title: {
                          text: 'Core Predictive Drivers (SHAP Value Contribution)',
                          font: { family: 'Outfit, sans-serif', size: 14, color: '#0F172A' },
                        },
                        autosize: true,
                        margin: { l: 180, r: 15, t: 35, b: 35 },
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        xaxis: {
                          title: { text: 'Influence (Log-Odds Contribution)', font: { size: 10, color: '#64748B' } },
                          tickfont: { size: 10, color: '#475569' },
                          gridcolor: '#F1F5F9',
                        },
                        yaxis: {
                          tickfont: { size: 10, color: '#0F172A' },
                          autorange: 'reversed',
                        },
                      }}
                      config={{ responsive: true, displayModeBar: false }}
                      style={{ width: '100%', height: '100%' }}
                    />
                  </div>
                </div>

                {/* Attribution Summary (5 cols) */}
                <div className="lg:col-span-5 bg-slate-50 p-5 rounded-xl border border-slate-200 text-xs space-y-2">
                  <h5 className="font-bold text-sm text-slate-900 border-b border-slate-200 pb-2">
                    Forecast Attribution Summary
                  </h5>
                  <ul className="space-y-1.5 text-slate-700 list-disc pl-4">
                    <li>
                      Raw Ensemble Probability: <b>{(shapData.predicted_probability * 100).toFixed(1)}%</b>
                    </li>
                    <li>
                      Model Forecast Decision: <b>{shapData.predicted_winner}</b>
                    </li>
                    {shapData.actual_outcome && (
                      <li>
                        Actual Outcome: <code className="bg-slate-200 px-1 rounded text-slate-900 font-semibold">{shapData.actual_outcome}</code>
                      </li>
                    )}
                    <li>
                      Regime Type: <b>{shapData.is_clean ? 'Democracy (Seat Share Model)' : 'Unclean (Classification)'}</b>
                    </li>
                    <li>
                      Active Data Sources: <b>V-Dem Index, WDI Macroeconomics, GDELT News Tone, Spatial Contagion</b>
                    </li>
                  </ul>
                </div>
              </div>

              {/* Detailed Feature Explanation Table */}
              <div className="space-y-2">
                <h5 className="font-bold text-sm text-slate-900">Detailed Feature Explanation Table</h5>
                <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
                  <table className="w-full text-left border-collapse text-xs">
                    <thead>
                      <tr className="bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold">
                        <th className="py-2.5 px-4">Feature Name</th>
                        <th className="py-2.5 px-4">Observed Value</th>
                        <th className="py-2.5 px-4">Influence Direction</th>
                        <th className="py-2.5 px-4 text-right">Influence Score</th>
                        <th className="py-2.5 px-4">Theoretical Explanation</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 text-slate-700">
                      {shapData.top_features.map((feat, idx) => (
                        <tr key={idx} className="hover:bg-slate-50">
                          <td className="py-2.5 px-4 font-semibold text-slate-900">{feat.human_name}</td>
                          <td className="py-2.5 px-4 font-mono">
                            {feat.value !== null ? feat.value.toFixed(2) : 'N/A'}
                          </td>
                          <td className="py-2.5 px-4">
                            <span
                              className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
                                feat.direction === 'Favors Incumbent'
                                  ? 'bg-blue-100 text-blue-700'
                                  : 'bg-orange-100 text-orange-700'
                              }`}
                            >
                              {feat.direction}
                            </span>
                          </td>
                          <td className="py-2.5 px-4 text-right font-mono font-bold">
                            {feat.contribution > 0 ? `+${feat.contribution}` : feat.contribution}
                          </td>
                          <td className="py-2.5 px-4 text-slate-600 leading-relaxed">{feat.literature_info}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ) : null}
        </div>
      ) : (
        <div className="p-4 rounded-xl bg-blue-50 border border-blue-200 text-blue-700 text-xs font-medium">
          💡 Select an election row in the tabs above to view its detailed SHAP drivers and theoretical breakdowns.
        </div>
      )}
    </div>
  );
};
