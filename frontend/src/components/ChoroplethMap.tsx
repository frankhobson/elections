import React, { useState } from 'react';
import type { UpcomingElection } from '../services/api';
import { MapPin, Calendar } from 'lucide-react';
import Plot from 'react-plotly.js';

interface ChoroplethMapProps {
  elections: UpcomingElection[];
  onSelectElection: (id: string) => void;
}

export const ChoroplethMap: React.FC<ChoroplethMapProps> = ({ elections, onSelectElection }) => {
  const [selectedYear, setSelectedYear] = useState<number>(2025);

  const yearElections = elections.filter((e) => e.year === selectedYear);

  // Prepare map arrays
  const countryCodes = yearElections.map((e) => e.country_code);
  const probabilities = yearElections.map((e) => e.raw_probability);
  const hoverTexts = yearElections.map(
    (e) =>
      `<b>${e.country_name} (${e.country_code})</b><br>` +
      `Election Year: ${e.year}<br>` +
      `Type: ${e.election_type}<br>` +
      `Favored: ${e.predicted_winner}<br>` +
      `Incumbent Win Prob: ${(e.raw_probability * 100).toFixed(1)}%<br>` +
      `Confidence: ${(e.adjusted_confidence * 100).toFixed(1)}%`
  );

  return (
    <div className="max-w-7xl mx-auto px-6 space-y-6 pb-12">
      {/* Map Header Controls */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800/80 flex flex-col md:flex-row items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2 m-0">
            <MapPin className="w-5 h-5 text-brandOrange" /> Global Forecast Map
          </h2>
          <p className="text-xs text-slate-400 mt-1 m-0">
            Choropleth map color-coded by predicted incumbent win probability (0.0 = Challenger, 1.0 = Incumbent)
          </p>
        </div>

        {/* Year Filter Buttons */}
        <div className="flex items-center gap-2 bg-slate-900/80 p-1.5 rounded-xl border border-slate-800">
          <span className="text-xs text-slate-400 px-2 flex items-center gap-1">
            <Calendar className="w-3.5 h-3.5" /> Forecast Cycle:
          </span>
          {[2025, 2026, 2027, 2028].map((year) => (
            <button
              key={year}
              onClick={() => setSelectedYear(year)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                selectedYear === year
                  ? 'bg-brandOrange text-white shadow-md shadow-brandOrange/20'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              {year}
            </button>
          ))}
        </div>
      </div>

      {/* Plotly Map Container */}
      <div className="glass-panel p-4 rounded-2xl border border-slate-800/80">
        <div className="w-full h-[540px]">
          <Plot
            data={[
              {
                type: 'choropleth',
                locationmode: 'ISO-3',
                locations: countryCodes,
                z: probabilities,
                text: hoverTexts,
                hoverinfo: 'text',
                colorscale: [
                  [0.0, '#F26419'], // Challenger (Orange)
                  [0.5, '#4B5563'], // Neutral / Toss-up
                  [1.0, '#10B981'], // Incumbent (Emerald)
                ],
                zmin: 0.0,
                zmax: 1.0,
                colorbar: {
                  title: { text: 'Incumbent Win Prob', font: { color: '#F8FAFC', size: 11 } },
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
            onClick={(event: any) => {
              if (event.points && event.points[0]) {
                const pointIdx = event.points[0].pointIndex;
                const match = yearElections[pointIdx];
                if (match) {
                  onSelectElection(match.election_id);
                }
              }
            }}
            style={{ width: '100%', height: '100%' }}
          />
        </div>
      </div>
    </div>
  );
};
