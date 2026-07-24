import React, { useState } from 'react';
import type { UpcomingElection } from '../services/api';
import Plot from 'react-plotly.js';

interface MapSectionProps {
  elections: UpcomingElection[];
  onSelectElection: (id: string) => void;
}

export const MapSection: React.FC<MapSectionProps> = ({ elections, onSelectElection }) => {
  const [mapFilterOpt, setMapFilterOpt] = useState<string>('All Upcoming (2026-2028)');

  let filtered = elections;
  if (mapFilterOpt !== 'All Upcoming (2026-2028)') {
    const tgtYear = parseInt(mapFilterOpt.split(' ')[0], 10);
    filtered = elections.filter((e) => e.year === tgtYear);
  }

  const locations = filtered.map((e) => e.country_code);
  const probabilities = filtered.map((e) => e.raw_probability);
  const hoverNames = filtered.map((e) => e.country_name);
  const customData = filtered.map((e) => [
    e.year,
    e.election_type,
    e.predicted_winner,
    `${(e.adjusted_confidence * 100).toFixed(1)}%`,
  ]);

  return (
    <div id="map-section" className="section-container">
      <h3 className="section-header">🗺️ Global Forecasts Choropleth Map</h3>
      <div className="section-subheader">
        Hover over countries to see forecast details. Blue represents predicted Incumbent victory, Red/Orange represents Challenger victory, and Gray indicates toss-up or unmodeled regions. Use the selector to inspect different forecasting cycles.
      </div>

      {/* Cycle Filter Selector */}
      <div className="mb-4 max-w-xs">
        <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1">
          Forecast Cycle Filter
        </label>
        <select
          value={mapFilterOpt}
          onChange={(e) => setMapFilterOpt(e.target.value)}
          className="w-full px-3 py-2 text-xs rounded-xl bg-slate-50 border border-slate-300 text-slate-900 font-medium focus:outline-none focus:border-blue-500 shadow-sm"
        >
          <option value="All Upcoming (2026-2028)">All Upcoming (2026-2028)</option>
          <option value="2026 Predictions">2026 Predictions</option>
          <option value="2027 Predictions">2027 Predictions</option>
          <option value="2028 Predictions">2028 Predictions</option>
        </select>
      </div>

      {/* Plotly Choropleth Map matching app.py */}
      <div className="w-full h-[480px]">
        <Plot
          data={[
            {
              type: 'choropleth',
              locationmode: 'ISO-3',
              locations: locations,
              z: probabilities,
              hovertext: hoverNames,
              customdata: customData,
              hovertemplate:
                '<b>%{hovertext}</b><br><br>' +
                'Year: <b>%{customdata[0]}</b><br>' +
                'Type: <b>%{customdata[1]}</b><br>' +
                'Predicted Winner: <b>%{customdata[2]}</b><br>' +
                'Incumbent Prob: <b>%{z:.1%}</b><br>' +
                'Confidence: <b>%{customdata[3]}</b><extra></extra>',
              colorscale: [
                [0.0, '#F26419'], // Challenger (Orange)
                [0.5, '#E2E8F0'], // Neutral
                [1.0, '#33658A'], // Incumbent (Blue)
              ],
              zmin: 0.0,
              zmax: 1.0,
              colorbar: {
                title: { text: 'Incumbent Win Prob', font: { family: 'Inter', size: 11, color: '#334155' } },
                tickformat: '.0%',
                ticks: 'outside',
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
              coastlinecolor: 'rgba(148,163,184,0.3)',
              bgcolor: 'rgba(0,0,0,0)',
              projection: {
                type: 'natural earth',
              },
            },
          }}
          config={{ responsive: true, displayModeBar: false }}
          onClick={(event: any) => {
            if (event.points && event.points[0]) {
              const idx = event.points[0].pointIndex;
              const match = filtered[idx];
              if (match) {
                onSelectElection(match.election_id);
                const el = document.getElementById('database-section');
                if (el) el.scrollIntoView({ behavior: 'smooth' });
              }
            }
          }}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    </div>
  );
};
