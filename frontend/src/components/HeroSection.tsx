import React from 'react';
import type { StatsResponse } from '../services/api';

interface HeroSectionProps {
  stats: StatsResponse | null;
}

export const HeroSection: React.FC<HeroSectionProps> = ({ stats }) => {
  const cvAccuracyStr = stats ? `${stats.overall_accuracy_pct.toFixed(2)}%` : '67.71%';
  const baselineAccuracy = 58.25;
  const gainVal = stats ? (stats.overall_accuracy_pct - baselineAccuracy).toFixed(2) : '9.46';
  const electionsCount = stats ? stats.historical_elections.toLocaleString() : '2,425';
  const activeProjections = stats ? stats.upcoming_elections.toLocaleString() : '158';

  return (
    <div id="summary" className="mb-10">
      {/* Hero Container */}
      <div className="hero-container">
        <div className="hero-title">🗳️ Global Election Forecaster</div>
        <div className="hero-subtitle">
          <p className="mb-3">
            <b>Using Global Indicators to Predict National Elections</b><br />
            This project explores how global and international macro-dynamics—rather than candidate-specific polling or demographics—can forecast national election outcomes.
            By training machine learning models on institutional V-Dem governance indices, global economic shocks, relative political momentum from GDELT, and spatial contagion lags,
            the forecaster maps international political shifts onto national races.
          </p>
          <p className="m-0">
            <b>Geopolitical Contagion &amp; Cascading Spillover Effects</b><br />
            Elections do not occur in isolation. To capture geopolitical diffusion, predictions are modeled as a chronological cascade:
            as elections are projected, their outcomes propagate trade, strategic alliance, and physical contiguity network weights to feed back
            into the features of neighboring nations' upcoming elections. An early change in one region can trigger cascading forecasts globally.
          </p>
        </div>
      </div>

      {/* Row of 4 KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="kpi-card">
          <div className="kpi-title">Model Strength</div>
          <div className="kpi-value text-slate-900">{cvAccuracyStr}</div>
          <div className="kpi-desc">Out-of-sample accuracy</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-title">Gain vs. Baseline</div>
          <div className="kpi-value text-blue-600">+{gainVal}%</div>
          <div className="kpi-desc">Above simple incumbent-wins rate ({baselineAccuracy}%)</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-title">Elections Modeled</div>
          <div className="kpi-value text-slate-900">{electionsCount}</div>
          <div className="kpi-desc">Historical records (1990-2026)</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-title">Active Predictions</div>
          <div className="kpi-value text-slate-900">{activeProjections}</div>
          <div className="kpi-desc">Upcoming forecasts through 2028</div>
        </div>
      </div>
    </div>
  );
};
