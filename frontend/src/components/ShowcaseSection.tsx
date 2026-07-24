import React from 'react';
import type { UpcomingElection } from '../services/api';
import Plot from 'react-plotly.js';

interface ShowcaseSectionProps {
  elections: UpcomingElection[];
  onSelectElection: (id: string) => void;
}

export const ShowcaseSection: React.FC<ShowcaseSectionProps> = ({ elections, onSelectElection }) => {
  const showcaseTargets = [
    { code: 'USA', year: 2028, type: 'Executive' },
    { code: 'FRA', year: 2027, type: 'Executive' },
    { code: 'DEU', year: 2027, type: 'Executive' },
    { code: 'MEX', year: 2027, type: 'Legislative' },
  ];

  return (
    <div className="mb-12">
      <h4 className="section-header">🔮 Prominent Upcoming Elections Showcase</h4>
      <div className="section-subheader">
        Key global elections forecasted by the cascading spatial GBDT models. Click on their entries in the Database Explorer below to review their dynamic feature drivers.
      </div>

      {/* 4 Columns Showcase Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        {showcaseTargets.map((tgt) => {
          const match = elections.find(
            (e) => e.country_code === tgt.code && e.year === tgt.year && e.election_type === tgt.type
          ) || elections.find((e) => e.country_code === tgt.code);

          if (!match) {
            return (
              <div key={tgt.code} className="showcase-card">
                <p className="text-xs text-slate-400">Loading forecast for {tgt.code}...</p>
              </div>
            );
          }

          const probPct = match.raw_probability * 100;
          const isIncumbent = match.predicted_winner === 'Incumbent';
          const barColor = isIncumbent ? '#33658A' : '#F26419';

          return (
            <div
              key={match.election_id}
              onClick={() => {
                onSelectElection(match.election_id);
                const el = document.getElementById('database-section');
                if (el) el.scrollIntoView({ behavior: 'smooth' });
              }}
              className="showcase-card"
            >
              <div className="showcase-country">🗳️ {match.country_name}</div>
              <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider my-1">
                {match.year} • {match.election_type}
              </div>

              <span className={`showcase-badge ${isIncumbent ? 'badge-incumbent' : 'badge-challenger'}`}>
                Projected: {match.predicted_winner}
              </span>

              {/* Plotly Dial Gauge */}
              <div className="w-full h-36 my-1">
                <Plot
                  data={[
                    {
                      type: 'indicator',
                      mode: 'gauge+number',
                      value: probPct,
                      number: {
                        suffix: '%',
                        font: { family: 'Outfit, sans-serif', size: 26, color: barColor, weight: 'bold' },
                      },
                      gauge: {
                        axis: { range: [0, 100], tickwidth: 1, tickcolor: '#475569', tickfont: { size: 10 } },
                        bar: { color: barColor, thickness: 0.6 },
                        bgcolor: 'rgba(0,0,0,0.05)',
                        borderwidth: 1,
                        bordercolor: '#cbd5e1',
                        steps: [
                          { range: [0, 50], color: 'rgba(242, 100, 25, 0.05)' },
                          { range: [50, 100], color: 'rgba(51, 101, 138, 0.05)' },
                        ],
                        threshold: {
                          line: { color: '#ef4444', width: 2 },
                          thickness: 0.8,
                          value: 50,
                        },
                      },
                    },
                  ]}
                  layout={{
                    autosize: true,
                    margin: { l: 15, r: 15, t: 10, b: 10 },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                  }}
                  config={{ responsive: true, displayModeBar: false }}
                  style={{ width: '100%', height: '100%' }}
                />
              </div>

              <div className="card-details">
                Confidence: <b>{(match.adjusted_confidence * 100).toFixed(1)}%</b><br />
                Data Quality: <b>{(match.data_completeness * 100).toFixed(1)}%</b>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
