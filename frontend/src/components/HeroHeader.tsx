import React from 'react';
import type { StatsResponse } from '../services/api';
import { Globe2, History, CheckCircle2, Calendar, ShieldCheck, Zap } from 'lucide-react';

interface HeroHeaderProps {
  stats: StatsResponse | null;
}

export const HeroHeader: React.FC<HeroHeaderProps> = ({ stats }) => {
  return (
    <div className="max-w-7xl mx-auto px-6 mb-8">
      {/* Metric Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {/* Card 1: Total Countries */}
        <div className="glass-card p-5 rounded-2xl flex items-center justify-between">
          <div>
            <p className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-1">Global Coverage</p>
            <h3 className="text-2xl font-extrabold text-white m-0">
              {stats ? stats.total_countries : '...'}
              <span className="text-xs font-normal text-slate-400 ml-1">countries</span>
            </h3>
          </div>
          <div className="w-12 h-12 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400">
            <Globe2 className="w-6 h-6" />
          </div>
        </div>

        {/* Card 2: Historical Dataset */}
        <div className="glass-card p-5 rounded-2xl flex items-center justify-between">
          <div>
            <p className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-1">Historical Archive</p>
            <h3 className="text-2xl font-extrabold text-white m-0">
              {stats ? stats.historical_elections : '...'}
              <span className="text-xs font-normal text-slate-400 ml-1">elections</span>
            </h3>
          </div>
          <div className="w-12 h-12 rounded-xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center text-purple-400">
            <History className="w-6 h-6" />
          </div>
        </div>

        {/* Card 3: Out-of-Sample Accuracy */}
        <div className="glass-card p-5 rounded-2xl flex items-center justify-between">
          <div>
            <p className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-1">Out-of-Sample Accuracy</p>
            <h3 className="text-2xl font-extrabold text-emerald-400 m-0">
              {stats ? `${stats.overall_accuracy_pct}%` : '...'}
              <span className="text-xs font-normal text-slate-400 ml-1.5">({stats ? `${stats.overall_correct}/${stats.historical_elections}` : ''})</span>
            </h3>
          </div>
          <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
            <CheckCircle2 className="w-6 h-6" />
          </div>
        </div>

        {/* Card 4: Upcoming Forecasts */}
        <div className="glass-card p-5 rounded-2xl flex items-center justify-between">
          <div>
            <p className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-1">Upcoming Forecasts</p>
            <h3 className="text-2xl font-extrabold text-brandOrange m-0">
              {stats ? stats.upcoming_elections : '...'}
              <span className="text-xs font-normal text-slate-400 ml-1">scheduled</span>
            </h3>
          </div>
          <div className="w-12 h-12 rounded-xl bg-orange-500/10 border border-orange-500/20 flex items-center justify-center text-brandOrange">
            <Calendar className="w-6 h-6" />
          </div>
        </div>
      </div>

      {/* Model Overview Banner */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800/80 bg-gradient-to-r from-slate-900/90 via-slate-900/60 to-slate-900/90">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm text-slate-300">
          <div className="flex items-start gap-3">
            <div className="p-2 rounded-lg bg-orange-500/10 text-orange-400 mt-0.5">
              <Zap className="w-4 h-4" />
            </div>
            <div>
              <h4 className="font-semibold text-white mb-1">Regime-Partitioned Hybrid Pipeline</h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                Democracies (<code className="text-orange-300 text-[11px]">clean_index ≥ 0.65</code>) are modeled using continuous seat-share regression mapped through an empirical Meta-Classifier. Unclean states retain high-capacity GBDT binary classifiers.
              </p>
            </div>
          </div>

          <div className="flex items-start gap-3">
            <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400 mt-0.5">
              <Globe2 className="w-4 h-4" />
            </div>
            <div>
              <h4 className="font-semibold text-white mb-1">Dynamic Cascading Network Lags</h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                Future election cycles propagate predictions sequentially into spatial trade/alliance networks and regional tides, updating global macroeconomic and democratic backsliding indicators dynamically.
              </p>
            </div>
          </div>

          <div className="flex items-start gap-3">
            <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400 mt-0.5">
              <ShieldCheck className="w-4 h-4" />
            </div>
            <div>
              <h4 className="font-semibold text-white mb-1">Out-of-Sample Rigor</h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                Evaluated via 5-fold cross-validation and chronological split folds (2010–2026). Zero post-election leakage with pre-election media tone and structural institutional fallbacks.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
