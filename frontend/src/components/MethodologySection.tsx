import React, { useState, useEffect } from 'react';
import type { VdemMapRecord, FeatureImportanceRecord } from '../services/api';
import { fetchVdemMap, fetchFeatureImportances } from '../services/api';
import Plot from 'react-plotly.js';

export const MethodologySection: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'pipeline' | 'vdem' | 'importance'>('pipeline');

  // Vdem Map state
  const [vdemYear, setVdemYear] = useState<number>(2020);
  const [vdemData, setVdemData] = useState<VdemMapRecord[]>([]);

  // Feature importances state
  const [importances, setImportances] = useState<FeatureImportanceRecord[]>([]);

  useEffect(() => {
    fetchVdemMap(vdemYear).then(setVdemData).catch(console.error);
  }, [vdemYear]);

  useEffect(() => {
    fetchFeatureImportances().then(setImportances).catch(console.error);
  }, []);

  return (
    <div id="methodology-section" className="section-container space-y-6">
      <h3 className="section-header">💡 Model Methodology &amp; Explainer</h3>
      <div className="section-subheader">
        Details on features, V-Dem indicators map, and feature importances.
      </div>

      {/* Tabs */}
      <div className="border-b border-slate-200 flex items-center gap-6">
        <button
          onClick={() => setActiveTab('pipeline')}
          className={`py-2.5 font-bold text-sm border-b-2 transition ${
            activeTab === 'pipeline'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          ⚙️ The Modeling Pipeline
        </button>
        <button
          onClick={() => setActiveTab('vdem')}
          className={`py-2.5 font-bold text-sm border-b-2 transition ${
            activeTab === 'vdem'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          🗺️ Historical V-Dem Cleanliness Map
        </button>
        <button
          onClick={() => setActiveTab('importance')}
          className={`py-2.5 font-bold text-sm border-b-2 transition ${
            activeTab === 'importance'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          🔥 Feature Importance Breakdowns
        </button>
      </div>

      {/* Tab 1: Pipeline */}
      {activeTab === 'pipeline' && (
        <div className="space-y-4 text-xs text-slate-700 leading-relaxed max-w-4xl">
          <h4 className="font-bold text-base text-slate-900 m-0">How it Works</h4>
          <p className="m-0">
            The Global Election Forecaster processes raw data through a four-stage modeling cascade:
          </p>
          <ol className="list-decimal pl-5 space-y-2">
            <li>
              <b>Data Ingestion</b>: Standardizes indicators from four distinct sources:
              <ul className="list-disc pl-5 mt-1 space-y-1">
                <li><b>V-Dem</b>: Structural governance, civil liberties, and institutional constraints.</li>
                <li><b>World Development Indicators (WDI)</b>: Macroeconomic levels and growth rates.</li>
                <li><b>GDELT Project</b>: Real-time media tone and protest event volume.</li>
                <li><b>CLEA Database</b>: Historical margins of victory and electoral systems.</li>
              </ul>
            </li>
            <li>
              <b>Network Contagion (Spatial Autoregression)</b>: Computes geographic, trade-based, and strategic alliance connectivity matrix weights to construct spatial lags, modeling the international diffusion of democracy, protests, and election outcomes.
            </li>
            <li>
              <b>GBDT Ensemble &amp; Meta-Classifier</b>: Trains separate XGBoost and CatBoost classifiers for <b>Executive</b> and <b>Legislative</b> elections. For democracies (<code className="bg-slate-100 px-1 rounded text-slate-800">clean_index ≥ 0.65</code>), seat shares are continuously regressed and mapped through an out-of-sample Logistic Regression Meta-Classifier to approximate Minimum Winning Coalitions (MWC).
            </li>
            <li>
              <b>Cascading Time-Series Forecasts</b>: For upcoming elections (2026-2028), the model forecasts chronologically. Each country's predicted outcome is fed-forward into subsequent neighboring predictions, adjusting the spatial lag features dynamically.
            </li>
          </ol>
        </div>
      )}

      {/* Tab 2: V-Dem Map */}
      {activeTab === 'vdem' && (
        <div className="space-y-4">
          <h5 className="font-bold text-sm text-slate-900 m-0">Historical Governance Explorer</h5>
          <p className="text-xs text-slate-600 m-0">
            Use the slider to explore the historical V-Dem Clean Elections Index dynamically across different years. This structural cleanliness forms the baseline layer of the GBDT models.
          </p>

          <div className="flex items-center gap-4 max-w-md bg-slate-50 p-3 rounded-xl border border-slate-200">
            <span className="text-xs font-bold text-slate-900 whitespace-nowrap">Year: {vdemYear}</span>
            <input
              type="range"
              min={1990}
              max={2025}
              value={vdemYear}
              onChange={(e) => setVdemYear(parseInt(e.target.value, 10))}
              className="w-full accent-blue-600 cursor-pointer"
            />
          </div>

          <div className="w-full h-[420px] bg-white rounded-xl border border-slate-200">
            <Plot
              data={[
                {
                  type: 'choropleth',
                  locationmode: 'ISO-3',
                  locations: vdemData.map((d) => d.country_code),
                  z: vdemData.map((d) => (d.clean_elections_index !== null ? d.clean_elections_index : 0)),
                  hovertext: vdemData.map((d) => d.country_name),
                  customdata: vdemData.map((d) => [
                    d.regime_name,
                    d.clean_elections_index !== null ? d.clean_elections_index.toFixed(3) : 'N/A',
                    d.polyarchy_index !== null ? d.polyarchy_index.toFixed(3) : 'N/A',
                  ]),
                  hovertemplate:
                    '<b>%{hovertext}</b><br><br>' +
                    'Regime Type: <b>%{customdata[0]}</b><br>' +
                    'Clean Elections Index: <b>%{customdata[1]}</b><br>' +
                    'Polyarchy Index: <b>%{customdata[2]}</b><extra></extra>',
                  colorscale: 'RdYlGn',
                  zmin: 0.0,
                  zmax: 1.0,
                  colorbar: {
                    title: { text: 'Cleanliness', font: { size: 11, color: '#334155' } },
                    tickformat: '.2f',
                    len: 0.8,
                  },
                },
              ]}
              layout={{
                autosize: true,
                margin: { l: 0, r: 0, t: 10, b: 0 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                geo: {
                  showframe: false,
                  showcoastlines: true,
                  coastlinecolor: '#86BBD8',
                  bgcolor: 'rgba(0,0,0,0)',
                  projection: { type: 'natural earth' },
                },
              }}
              config={{ responsive: true, displayModeBar: false }}
              style={{ width: '100%', height: '100%' }}
            />
          </div>
        </div>
      )}

      {/* Tab 3: Feature Importance */}
      {activeTab === 'importance' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-5 bg-white p-4 rounded-xl border border-slate-200">
            <h5 className="font-bold text-sm text-slate-900 mb-3">Top 10 Model Predictors</h5>
            <div className="w-full h-80">
              <Plot
                data={[
                  {
                    type: 'bar',
                    orientation: 'h',
                    x: importances.map((d) => d.importance),
                    y: importances.map((d) => d.human_name),
                    marker: {
                      color: importances.map((_, i) => ['#86BBD8', '#33658A', '#2F4858'][i % 3]),
                    },
                  },
                ]}
                layout={{
                  autosize: true,
                  margin: { l: 160, r: 15, t: 10, b: 35 },
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  xaxis: { title: { text: 'Importance Score', font: { size: 10 } } },
                  yaxis: { autorange: 'reversed', tickfont: { size: 10 } },
                }}
                config={{ responsive: true, displayModeBar: false }}
                style={{ width: '100%', height: '100%' }}
              />
            </div>
          </div>

          <div className="lg:col-span-7 space-y-3">
            <h5 className="font-bold text-sm text-slate-900 m-0">Theoretical Explanations</h5>
            <p className="text-xs text-slate-600 m-0">
              Below are descriptions of the core theoretical features driving GBDT outcomes:
            </p>

            <div className="divide-y divide-slate-200 border border-slate-200 rounded-xl overflow-hidden bg-slate-50">
              {importances.map((item, idx) => (
                <details key={idx} className="group p-3 cursor-pointer">
                  <summary className="font-semibold text-xs text-slate-900 flex items-center justify-between">
                    <span>Rank {idx + 1}: {item.human_name}</span>
                    <span className="text-slate-400 group-open:rotate-180 transition">▼</span>
                  </summary>
                  <p className="text-xs text-slate-600 mt-2 leading-relaxed m-0">{item.explanation}</p>
                </details>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
