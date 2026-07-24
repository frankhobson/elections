import React, { useEffect, useState } from 'react';
import type { ShapResponse } from '../services/api';
import { fetchShapExplanation } from '../services/api';
import { X, Loader2, Info, BookOpen } from 'lucide-react';
import Plot from 'react-plotly.js';

interface ShapModalProps {
  electionId: string | null;
  onClose: () => void;
}

export const ShapModal: React.FC<ShapModalProps> = ({ electionId, onClose }) => {
  const [data, setData] = useState<ShapResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!electionId) return;

    setLoading(true);
    setError(null);
    fetchShapExplanation(electionId)
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setError('Failed to load feature attribution data.');
        setLoading(false);
      });
  }, [electionId]);

  if (!electionId) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-fadeIn">
      <div className="glass-panel w-full max-w-4xl max-h-[90vh] rounded-2xl border border-slate-700/80 shadow-2xl flex flex-col overflow-hidden">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/60">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-orange-500/20 border border-orange-500/30 flex items-center justify-center text-orange-400 font-semibold text-xs">
              {data ? data.country_code : '...'}
            </div>
            <div>
              <h3 className="text-base font-bold text-white m-0">
                {data ? `${data.country_code} ${data.year} (${data.election_type})` : 'Loading...'}
              </h3>
              <p className="text-xs text-slate-400 m-0">Feature Driver Attributions (SHAP Values)</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto flex-1 space-y-6">
          {loading && (
            <div className="flex flex-col items-center justify-center py-16 text-slate-400 gap-3">
              <Loader2 className="w-8 h-8 animate-spin text-brandOrange" />
              <p className="text-sm">Calculating SHAP feature drivers...</p>
            </div>
          )}

          {error && (
            <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-300 text-sm">
              {error}
            </div>
          )}

          {data && !loading && (
            <>
              {/* Summary Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800">
                  <p className="text-xs text-slate-400 mb-1">Forecast Winner</p>
                  <p className="text-lg font-bold text-white flex items-center gap-2">
                    {data.predicted_winner}
                    <span className="text-xs font-normal text-orange-400">
                      ({(data.predicted_probability * 100).toFixed(1)}%)
                    </span>
                  </p>
                </div>

                <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800">
                  <p className="text-xs text-slate-400 mb-1">Regime Cleanliness</p>
                  <p className="text-lg font-bold text-white">
                    {data.is_clean ? (
                      <span className="text-emerald-400">Democracy (Seat Share)</span>
                    ) : (
                      <span className="text-amber-400">Unclean (Classification)</span>
                    )}
                  </p>
                </div>

                <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800">
                  <p className="text-xs text-slate-400 mb-1">Actual Outcome</p>
                  <p className="text-lg font-bold text-slate-200">
                    {data.actual_outcome ? data.actual_outcome : 'Upcoming / Unknown'}
                  </p>
                </div>
              </div>

              {/* Horizontal Bar Chart */}
              <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800">
                <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                  <Info className="w-4 h-4 text-brandCyan" /> Core Predictive Drivers (Influence)
                </h4>
                <div className="w-full h-80">
                  <Plot
                    data={[
                      {
                        type: 'bar',
                        orientation: 'h',
                        x: data.top_features.map((f) => f.contribution),
                        y: data.top_features.map((f) => f.human_name),
                        marker: {
                          color: data.top_features.map((f) =>
                            f.direction === 'Favors Incumbent' ? '#33658A' : '#F26419'
                          ),
                        },
                      },
                    ]}
                    layout={{
                      autosize: true,
                      margin: { l: 180, r: 20, t: 10, b: 40 },
                      paper_bgcolor: 'rgba(0,0,0,0)',
                      plot_bgcolor: 'rgba(0,0,0,0)',
                      xaxis: {
                        title: { text: 'Impact (Log-Odds Contribution)', font: { color: '#94A3B8', size: 11 } },
                        tickfont: { color: '#94A3B8', size: 10 },
                        gridcolor: '#2D3748',
                      },
                      yaxis: {
                        tickfont: { color: '#F8FAFC', size: 11 },
                        autorange: 'reversed',
                      },
                    }}
                    config={{ responsive: true, displayModeBar: false }}
                    style={{ width: '100%', height: '100%' }}
                  />
                </div>
              </div>

              {/* Theoretical Explanations List */}
              <div className="space-y-3">
                <h4 className="text-sm font-semibold text-white flex items-center gap-2">
                  <BookOpen className="w-4 h-4 text-orange-400" /> Political Science Feature Literature &amp; Values
                </h4>
                <div className="divide-y divide-slate-800 border border-slate-800 rounded-xl overflow-hidden bg-slate-900/40">
                  {data.top_features.map((feat, idx) => (
                    <div key={idx} className="p-3.5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-semibold text-white">{feat.human_name}</span>
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-medium ${
                              feat.direction === 'Favors Incumbent'
                                ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                                : 'bg-orange-500/10 text-orange-400 border border-orange-500/20'
                            }`}
                          >
                            {feat.direction}
                          </span>
                        </div>
                        <p className="text-slate-400 leading-normal m-0">{feat.literature_info}</p>
                      </div>
                      <div className="text-right flex sm:flex-col items-center sm:items-end justify-between w-full sm:w-auto gap-2">
                        <span className="font-mono text-slate-300">
                          Val: {feat.value !== null ? feat.value.toFixed(2) : 'N/A'}
                        </span>
                        <span className={`font-bold font-mono ${feat.contribution > 0 ? 'text-blue-400' : 'text-orange-400'}`}>
                          {feat.contribution > 0 ? `+${feat.contribution}` : feat.contribution}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
