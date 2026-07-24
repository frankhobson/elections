import React, { useState } from 'react';
import type { DiagnosticsResponse } from '../services/api';
import { BarChart3, CheckCircle2, Sliders, Layers } from 'lucide-react';
import Plot from 'react-plotly.js';

interface ModelDiagnosticsProps {
  diagnostics: DiagnosticsResponse | null;
}

export const ModelDiagnostics: React.FC<ModelDiagnosticsProps> = ({ diagnostics }) => {
  const [activeTab, setActiveTab] = useState<'cv' | 'dist'>('cv');

  if (!diagnostics) {
    return (
      <div className="max-w-7xl mx-auto px-6 py-12 text-center text-slate-500">
        Loading model diagnostics...
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-6 space-y-6 pb-12">
      {/* Diagnostics Header */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800/80 flex flex-col md:flex-row items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2 m-0">
            <BarChart3 className="w-5 h-5 text-brandOrange" /> Model Performance Diagnostics &amp; Cross-Validation
          </h2>
          <p className="text-xs text-slate-400 mt-1 m-0">
            Chronological out-of-sample split evaluations and country accuracy distribution histograms.
          </p>
        </div>

        {/* View Switcher */}
        <div className="flex items-center gap-1 bg-slate-900/80 p-1 rounded-xl border border-slate-800">
          <button
            onClick={() => setActiveTab('cv')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition flex items-center gap-1.5 ${
              activeTab === 'cv' ? 'bg-brandOrange text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            <Layers className="w-3.5 h-3.5" /> Chronological Folds
          </button>
          <button
            onClick={() => setActiveTab('dist')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition flex items-center gap-1.5 ${
              activeTab === 'dist' ? 'bg-brandOrange text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" /> Country Distribution
          </button>
        </div>
      </div>

      {/* Model Performance Averages Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="glass-card p-5 rounded-2xl border border-slate-800 flex items-center justify-between">
          <div>
            <p className="text-xs text-slate-400 font-medium mb-1">CatBoost CV Mean</p>
            <h3 className="text-2xl font-extrabold text-white m-0 font-mono">
              {diagnostics.mean_catboost.toFixed(2)}%
            </h3>
          </div>
          <div className="p-3 rounded-xl bg-purple-500/10 text-purple-400 border border-purple-500/20">
            <CheckCircle2 className="w-5 h-5" />
          </div>
        </div>

        <div className="glass-card p-5 rounded-2xl border border-slate-800 flex items-center justify-between">
          <div>
            <p className="text-xs text-slate-400 font-medium mb-1">XGBoost CV Mean</p>
            <h3 className="text-2xl font-extrabold text-white m-0 font-mono">
              {diagnostics.mean_xgboost.toFixed(2)}%
            </h3>
          </div>
          <div className="p-3 rounded-xl bg-blue-500/10 text-blue-400 border border-blue-500/20">
            <CheckCircle2 className="w-5 h-5" />
          </div>
        </div>

        <div className="glass-card p-5 rounded-2xl border border-slate-800 flex items-center justify-between">
          <div>
            <p className="text-xs text-slate-400 font-medium mb-1">Voting Ensemble CV Mean</p>
            <h3 className="text-2xl font-extrabold text-brandOrange m-0 font-mono">
              {diagnostics.mean_ensemble.toFixed(2)}%
            </h3>
          </div>
          <div className="p-3 rounded-xl bg-orange-500/10 text-brandOrange border border-orange-500/20">
            <CheckCircle2 className="w-5 h-5" />
          </div>
        </div>
      </div>

      {/* Tab 1: Chronological CV Table */}
      {activeTab === 'cv' && (
        <div className="glass-panel p-6 rounded-2xl border border-slate-800/80 space-y-4">
          <h3 className="text-sm font-bold text-white mb-2">Rolling Chronological Cross-Validation Folds</h3>
          <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-900/40">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-800 bg-slate-900/80 text-slate-400 font-medium">
                  <th className="py-3 px-4">Training Cutoff &amp; Test Period</th>
                  <th className="py-3 px-4 text-center">CatBoost Classifier</th>
                  <th className="py-3 px-4 text-center">XGBoost Classifier</th>
                  <th className="py-3 px-4 text-right">Voting Ensemble</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-200">
                {diagnostics.cv_folds.map((fold, idx) => (
                  <tr key={idx} className="hover:bg-slate-800/40 transition">
                    <td className="py-3 px-4 font-bold text-white font-mono">{fold.Origin}</td>
                    <td className="py-3 px-4 text-center font-mono text-purple-300">
                      {(fold['CatBoost Classifier'] * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-4 text-center font-mono text-blue-300">
                      {(fold['XGBoost Classifier'] * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-4 text-right font-mono font-bold text-emerald-400">
                      {(fold['Voting Ensemble'] * 100).toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab 2: Country Accuracy Distribution Histogram */}
      {activeTab === 'dist' && (
        <div className="glass-panel p-6 rounded-2xl border border-slate-800/80 space-y-4">
          <h3 className="text-sm font-bold text-white mb-2">Country Accuracy Distribution (168 Countries)</h3>
          <div className="w-full h-80">
            <Plot
              data={[
                {
                  type: 'bar',
                  x: diagnostics.country_accuracy_distribution.map((d) => d.bin),
                  y: diagnostics.country_accuracy_distribution.map((d) => d.count),
                  marker: {
                    color: '#E05A1A',
                    line: { color: '#1E222A', width: 1.5 },
                  },
                },
              ]}
              layout={{
                autosize: true,
                margin: { l: 40, r: 20, t: 20, b: 40 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                xaxis: {
                  title: { text: 'Accuracy Bucket (%)', font: { color: '#94A3B8', size: 11 } },
                  tickfont: { color: '#F8FAFC', size: 11 },
                  gridcolor: '#2D3748',
                },
                yaxis: {
                  title: { text: 'Number of Countries', font: { color: '#94A3B8', size: 11 } },
                  tickfont: { color: '#94A3B8', size: 11 },
                  gridcolor: '#2D3748',
                },
              }}
              config={{ responsive: true, displayModeBar: false }}
              style={{ width: '100%', height: '100%' }}
            />
          </div>
        </div>
      )}
    </div>
  );
};
