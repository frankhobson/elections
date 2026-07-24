import React, { useState } from 'react';
import type { DiagnosticsResponse, CountryAccuracy } from '../services/api';
import Plot from 'react-plotly.js';

interface DiagnosticsSectionProps {
  diagnostics: DiagnosticsResponse | null;
  countryAccuracy: CountryAccuracy[];
}

export const DiagnosticsSection: React.FC<DiagnosticsSectionProps> = ({ diagnostics, countryAccuracy }) => {
  const [activeTab, setActiveTab] = useState<'split' | 'calibration' | 'confusion' | 'country'>('split');

  if (!diagnostics) {
    return (
      <div id="diagnostics-section" className="section-container">
        <h3 className="section-header">⚙️ Model Diagnostics &amp; Evaluation Metrics</h3>
        <p className="text-xs text-slate-500">Loading diagnostics data...</p>
      </div>
    );
  }

  // Country accuracy map data (filter <3 total elections for color scaling)
  const validCountries = countryAccuracy.filter((c) => c.total >= 3);
  const grayCountries = countryAccuracy.filter((c) => c.total < 3);

  const calibrationBins = diagnostics.calibration_bins || [];
  const cmExec = diagnostics.confusion_matrix_exec || { tp: 0, fp: 0, fn: 0, tn: 0 };
  const cmLeg = diagnostics.confusion_matrix_leg || { tp: 0, fp: 0, fn: 0, tn: 0 };

  return (
    <div id="diagnostics-section" className="section-container space-y-6">
      <h3 className="section-header">⚙️ Model Diagnostics &amp; Evaluation Metrics</h3>
      <div className="section-subheader">
        Review cross-validation metrics, confusion matrices, and probability calibration.
      </div>

      {/* 4 Streamlit Tabs */}
      <div className="border-b border-slate-200 flex items-center gap-6 overflow-x-auto">
        <button
          onClick={() => setActiveTab('split')}
          className={`py-2.5 font-bold text-sm border-b-2 whitespace-nowrap transition ${
            activeTab === 'split'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          📊 Rolling Split Performance
        </button>
        <button
          onClick={() => setActiveTab('calibration')}
          className={`py-2.5 font-bold text-sm border-b-2 whitespace-nowrap transition ${
            activeTab === 'calibration'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          📈 Probability Calibration
        </button>
        <button
          onClick={() => setActiveTab('confusion')}
          className={`py-2.5 font-bold text-sm border-b-2 whitespace-nowrap transition ${
            activeTab === 'confusion'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          🎛️ Confusion Matrices &amp; Hyperparameters
        </button>
        <button
          onClick={() => setActiveTab('country')}
          className={`py-2.5 font-bold text-sm border-b-2 whitespace-nowrap transition ${
            activeTab === 'country'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          🗺️ Accuracy by Country
        </button>
      </div>

      {/* Tab 1: Rolling Split Performance */}
      {activeTab === 'split' && (
        <div className="space-y-6">
          {/* Executive / Legislative / Combined Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 text-center">
              <div className="bg-[#3A75A2] text-white font-bold py-1 px-3 rounded text-xs mb-3">
                Combined Metrics (All)
              </div>
              <div className="text-xl font-bold text-slate-900 font-mono">67.71%</div>
              <p className="text-[11px] text-slate-500">OOS Accuracy (5-Fold CV)</p>
            </div>

            <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 text-center">
              <div className="bg-[#E05A1A] text-white font-bold py-1 px-3 rounded text-xs mb-3">
                Executive
              </div>
              <div className="text-xl font-bold text-slate-900 font-mono">68.99%</div>
              <p className="text-[11px] text-slate-500">OOS Accuracy (5-Fold CV)</p>
            </div>

            <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 text-center">
              <div className="bg-[#4A8EB6] text-white font-bold py-1 px-3 rounded text-xs mb-3">
                Legislative (Congressional)
              </div>
              <div className="text-xl font-bold text-slate-900 font-mono">66.95%</div>
              <p className="text-[11px] text-slate-500">OOS Accuracy (5-Fold CV)</p>
            </div>
          </div>

          <div>
            <h5 className="font-bold text-sm text-slate-900 mb-1">
              Out-of-Sample Accuracy across Historical Validation Breakpoints
            </h5>
            <p className="text-xs text-slate-600 mb-3">
              By training models only on data before a specific milestone year (e.g. 2010) and validating on the subsequent 4 years (e.g. 2011-2014), we ensure our validation mimics true out-of-sample forecasting:
            </p>

            <div className="w-full h-80 bg-white p-3 rounded-xl border border-slate-200">
              <Plot
                data={[
                  {
                    type: 'bar',
                    name: 'CatBoost',
                    x: diagnostics.cv_folds.map((f) => f.Origin),
                    y: diagnostics.cv_folds.map((f) => f['CatBoost Classifier']),
                    marker: { color: '#86BBD8' },
                  },
                  {
                    type: 'bar',
                    name: 'XGBoost',
                    x: diagnostics.cv_folds.map((f) => f.Origin),
                    y: diagnostics.cv_folds.map((f) => f['XGBoost Classifier']),
                    marker: { color: '#33658A' },
                  },
                  {
                    type: 'bar',
                    name: 'Ensemble',
                    x: diagnostics.cv_folds.map((f) => f.Origin),
                    y: diagnostics.cv_folds.map((f) => f['Voting Ensemble']),
                    marker: { color: '#F6AE2D' },
                  },
                ]}
                layout={{
                  barmode: 'group',
                  autosize: true,
                  margin: { l: 40, r: 15, t: 30, b: 40 },
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  xaxis: { title: { text: 'Historical Training Cutoff & Test Period', font: { size: 11 } } },
                  yaxis: { title: { text: 'OOS Accuracy', font: { size: 11 } }, tickformat: '.1%', range: [0.4, 0.85] },
                  legend: { orientation: 'h', y: 1.15, x: 0 },
                }}
                config={{ responsive: true, displayModeBar: false }}
                style={{ width: '100%', height: '100%' }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Calibration Curve */}
      {activeTab === 'calibration' && (
        <div className="space-y-4">
          <h5 className="font-bold text-sm text-slate-900 m-0">Probability Calibration Curve</h5>
          <p className="text-xs text-slate-600 m-0">
            A well-calibrated forecast is one where a predicted win probability of 70% translates to the incumbent winning approximately 70% of the time. The chart below checks calibrated probabilities against actual win rates:
          </p>

          <div className="w-full h-80 bg-white p-4 rounded-xl border border-slate-200">
            <Plot
              data={[
                // Bars for sample sizes
                {
                  type: 'bar',
                  name: 'Exec Count (Right Axis)',
                  x: calibrationBins.map((b) => b.bin),
                  y: calibrationBins.map((b) => b.exec_count),
                  yaxis: 'y2',
                  marker: { color: '#F26419' },
                  opacity: 0.25,
                },
                {
                  type: 'bar',
                  name: 'Leg Count (Right Axis)',
                  x: calibrationBins.map((b) => b.bin),
                  y: calibrationBins.map((b) => b.leg_count),
                  yaxis: 'y2',
                  marker: { color: '#33658A' },
                  opacity: 0.25,
                },
                // Perfect calibration line (y=x)
                {
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Perfect Calibration (y=x)',
                  x: calibrationBins.map((b) => b.bin),
                  y: calibrationBins.map((b) => b.midpoint),
                  line: { color: '#F6AE2D', width: 2, dash: 'dash' },
                },
                // Real empirical Executive accuracy
                {
                  type: 'scatter',
                  mode: 'lines+markers',
                  name: 'Executive Accuracy',
                  x: calibrationBins.filter((b) => b.exec_accuracy !== null).map((b) => b.bin),
                  y: calibrationBins.filter((b) => b.exec_accuracy !== null).map((b) => b.exec_accuracy),
                  line: { color: '#F26419', width: 3 },
                  marker: { size: 8 },
                },
                // Real empirical Legislative accuracy
                {
                  type: 'scatter',
                  mode: 'lines+markers',
                  name: 'Legislative Accuracy',
                  x: calibrationBins.filter((b) => b.leg_accuracy !== null).map((b) => b.bin),
                  y: calibrationBins.filter((b) => b.leg_accuracy !== null).map((b) => b.leg_accuracy),
                  line: { color: '#33658A', width: 3 },
                  marker: { size: 8 },
                },
              ]}
              layout={{
                autosize: true,
                margin: { l: 40, r: 40, t: 30, b: 40 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                xaxis: { title: { text: 'Model Confidence Bin', font: { size: 11 } } },
                yaxis: { title: { text: 'Actual OOS Accuracy', font: { size: 11 } }, tickformat: '.0%', range: [0.4, 1.05] },
                yaxis2: {
                  title: { text: 'Sample Size', font: { size: 11 } },
                  overlaying: 'y',
                  side: 'right',
                  showgrid: false,
                },
                barmode: 'stack',
                legend: { orientation: 'h', y: 1.18, x: 0 },
              }}
              config={{ responsive: true, displayModeBar: false }}
              style={{ width: '100%', height: '100%' }}
            />
          </div>
        </div>
      )}

      {/* Tab 3: Confusion Matrices & Hyperparams */}
      {activeTab === 'confusion' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white p-4 rounded-xl border border-slate-200 space-y-4">
            <h5 className="font-bold text-sm text-slate-900 m-0">Executive Confusion Matrix</h5>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs mb-3">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-200">
                    <th className="py-2 px-3">Outcome</th>
                    <th className="py-2 px-3 text-center">Predicted Incumbent Victory</th>
                    <th className="py-2 px-3 text-center">Predicted Challenger Victory</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-slate-700">
                  <tr>
                    <td className="py-2 px-3 font-semibold">Actual Challenger Loss</td>
                    <td className="py-2 px-3 text-center font-mono font-bold text-emerald-600">{cmExec.tp}</td>
                    <td className="py-2 px-3 text-center font-mono text-slate-500">{cmExec.fn}</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 font-semibold">Actual Incumbent Loss</td>
                    <td className="py-2 px-3 text-center font-mono text-slate-500">{cmExec.fp}</td>
                    <td className="py-2 px-3 text-center font-mono font-bold text-emerald-600">{cmExec.tn}</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <h5 className="font-bold text-sm text-slate-900 m-0 pt-2 border-t border-slate-200">Legislative Confusion Matrix</h5>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-200">
                    <th className="py-2 px-3">Outcome</th>
                    <th className="py-2 px-3 text-center">Predicted Incumbent Victory</th>
                    <th className="py-2 px-3 text-center">Predicted Challenger Victory</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-slate-700">
                  <tr>
                    <td className="py-2 px-3 font-semibold">Actual Challenger Loss</td>
                    <td className="py-2 px-3 text-center font-mono font-bold text-emerald-600">{cmLeg.tp}</td>
                    <td className="py-2 px-3 text-center font-mono text-slate-500">{cmLeg.fn}</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 font-semibold">Actual Incumbent Loss</td>
                    <td className="py-2 px-3 text-center font-mono text-slate-500">{cmLeg.fp}</td>
                    <td className="py-2 px-3 text-center font-mono font-bold text-emerald-600">{cmLeg.tn}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 space-y-2">
            <h5 className="font-bold text-sm text-slate-900 m-0">Model Calibration Parameters &amp; Settings</h5>
            <p className="text-xs text-slate-600">
              Temperature-scaled probability adjustments and ensemble weights determined by validation grid searches:
            </p>
            <pre className="bg-slate-900 text-slate-200 p-3 rounded-lg text-[11px] font-mono overflow-x-auto">
              {JSON.stringify(
                {
                  clean_classification_threshold: 0.51,
                  exec_unclean_threshold: 0.53,
                  leg_unclean_threshold: 0.49,
                  executive_model: { xgb_weight: 0.7, T: 0.78 },
                  legislative_model: { xgb_weight: 0.9, T: 0.80 },
                  ensemble_balance: { T: 0.70 },
                },
                null,
                2
              )}
            </pre>
          </div>
        </div>
      )}

      {/* Tab 4: Accuracy by Country */}
      {activeTab === 'country' && (
        <div className="space-y-6">
          <div className="space-y-2">
            <h5 className="font-bold text-sm text-slate-900 m-0">Out-of-Sample Accuracy by Country</h5>
            <p className="text-xs text-slate-600 leading-relaxed m-0">
              This choropleth map visualizes the model's out-of-sample historical prediction accuracy for each country. To prevent countries with very few elections from distorting the color scale (e.g. showing 100% or 0% accuracy from only 1 or 2 races), countries with <b>fewer than 3 total elections are colored in gray</b>.
            </p>
          </div>

          {/* Choropleth Map with Red -> Yellow -> Green Color Scale */}
          <div className="w-full h-[450px]">
            <Plot
              data={[
                {
                  type: 'choropleth',
                  locationmode: 'ISO-3',
                  locations: validCountries.map((c) => c.country_code),
                  z: validCountries.map((c) => c.accuracy_pct / 100.0), // Decimal 0.4 to 1.0
                  hovertext: validCountries.map((c) => c.country_name),
                  customdata: validCountries.map((c) => [
                    `${c.accuracy_pct.toFixed(1)}%`,
                    `${c.exec_correct}/${c.exec_total}`,
                    `${c.leg_correct}/${c.leg_total}`,
                  ]),
                  hovertemplate:
                    '<b>%{hovertext}</b><br><br>' +
                    'Historical OOS Accuracy: <b>%{customdata[0]}</b><br>' +
                    'Executive: <b>%{customdata[1]}</b><br>' +
                    'Legislative: <b>%{customdata[2]}</b><extra></extra>',
                  colorscale: [
                    [0.0, '#dc2626'],  // Red for low accuracy (~40%)
                    [0.5, '#eab308'],  // Yellow for medium accuracy (~70%)
                    [1.0, '#16a34a'],  // Green for high accuracy (100%)
                  ],
                  zmin: 0.4,
                  zmax: 1.0,
                  colorbar: {
                    title: { text: 'Accuracy', font: { size: 11, color: '#334155' } },
                    tickformat: '.0%',
                  },
                },
                {
                  type: 'choropleth',
                  locationmode: 'ISO-3',
                  locations: grayCountries.map((c) => c.country_code),
                  z: grayCountries.map(() => 0.5),
                  colorscale: [
                    [0, '#cbd5e1'],
                    [1, '#cbd5e1'],
                  ],
                  showscale: false,
                  hovertext: grayCountries.map((c) => c.country_name),
                  customdata: grayCountries.map((c) => [
                    `${c.accuracy_pct.toFixed(1)}%`,
                    `${c.exec_correct}/${c.exec_total}`,
                    `${c.leg_correct}/${c.leg_total}`,
                  ]),
                  hovertemplate:
                    '<b>%{hovertext}</b><br><br>' +
                    'Historical OOS Accuracy: <b>%{customdata[0]}</b><br>' +
                    'Executive: <b>%{customdata[1]}</b><br>' +
                    'Legislative: <b>%{customdata[2]}</b><br>' +
                    '<i>(Fewer than 3 elections)</i><extra></extra>',
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

          {/* Ranked Country Table (Max Height 380px Scrollable Container) */}
          <div className="max-h-[380px] overflow-y-auto rounded-xl border border-slate-200 bg-white shadow-sm">
            <table className="w-full text-left border-collapse text-xs">
              <thead className="sticky top-0 z-10 bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold shadow-xs">
                <tr>
                  <th className="py-2.5 px-4 text-center">Rank</th>
                  <th className="py-2.5 px-4">Country Name</th>
                  <th className="py-2.5 px-4">Code</th>
                  <th className="py-2.5 px-4">Region</th>
                  <th className="py-2.5 px-4 text-center">Total Elections</th>
                  <th className="py-2.5 px-4 text-center">Correct Predictions</th>
                  <th className="py-2.5 px-4 text-center">Executive</th>
                  <th className="py-2.5 px-4 text-center">Legislative</th>
                  <th className="py-2.5 px-4 text-right">Accuracy Rate</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-slate-700">
                {countryAccuracy.map((c) => (
                  <tr key={c.country_code} className="hover:bg-slate-50">
                    <td className="py-2.5 px-4 text-center font-mono text-slate-400">#{c.rank}</td>
                    <td className="py-2.5 px-4 font-bold text-slate-900">{c.country_name}</td>
                    <td className="py-2.5 px-4 font-mono text-slate-500">{c.country_code}</td>
                    <td className="py-2.5 px-4 text-slate-600">{c.region}</td>
                    <td className="py-2.5 px-4 text-center font-mono">{c.total}</td>
                    <td className="py-2.5 px-4 text-center font-mono font-bold text-emerald-600">{c.correct}</td>
                    <td className="py-2.5 px-4 text-center font-mono text-slate-500">
                      {c.exec_correct}/{c.exec_total}
                    </td>
                    <td className="py-2.5 px-4 text-center font-mono text-slate-500">
                      {c.leg_correct}/{c.leg_total}
                    </td>
                    <td className="py-2.5 px-4 text-right font-mono font-extrabold text-slate-900">
                      {c.accuracy_pct.toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
