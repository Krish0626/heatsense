import React, { useState } from 'react';
import { PARETO_SOLUTIONS } from '../data/mockData';
import { ParetoSolution } from '../types';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
} from 'recharts';

export const ParetoExplorer: React.FC = () => {
  const [selectedSolution, setSelectedSolution] = useState<ParetoSolution>(PARETO_SOLUTIONS[1]);

  return (
    <div className="glass-panel" style={{ padding: '24px', height: '100%', overflowY: 'auto' }}>
      <div style={{ marginBottom: '18px' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 700, fontFamily: 'var(--font-display)', color: 'var(--text-primary)' }}>
          NSGA-II Multi-Objective Optimization (Pareto Front)
        </h2>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          Algorithmic trade-off between Municipal Budget (₹ Lakhs) and City-Wide Cooling (°C) across Surat hotspots
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '20px' }}>
        {/* Pareto Frontier Scatter Chart */}
        <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '16px', borderRadius: '12px', border: '1px solid var(--border-subtle)', height: '360px' }}>
          <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px' }}>
            Non-Dominated Solutions (Click a point to inspect)
          </div>
          <ResponsiveContainer width="100%" height="90%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
              <XAxis
                type="number"
                dataKey="total_cost_lakh_inr"
                name="Budget"
                unit=" L"
                stroke="#64748b"
                fontSize={12}
                label={{ value: 'Total 10-Yr Cost (₹ Lakhs)', position: 'insideBottom', offset: -12, fill: '#94a3b8', fontSize: 11 }}
              />
              <YAxis
                type="number"
                dataKey="total_cooling_c"
                name="Cooling"
                unit="°C"
                stroke="#64748b"
                fontSize={12}
                domain={[0.5, 5.0]}
                label={{ value: 'Total Cooling (°C)', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 11 }}
              />
              <Tooltip
                cursor={{ strokeDasharray: '3 3' }}
                content={({ active, payload }: any) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload as ParetoSolution;
                    return (
                      <div style={{ background: '#0f172a', border: '1px solid #38bdf8', padding: '10px', borderRadius: '8px', fontSize: '12px' }}>
                        <div style={{ fontWeight: 700, color: 'var(--accent-cyan)' }}>{data.name}</div>
                        <div>Cooling: <strong>-{data.total_cooling_c}°C</strong></div>
                        <div>Cost: <strong>₹{data.total_cost_lakh_inr} Lakhs</strong></div>
                        <div>Hotspots: <strong>{data.hotspots_addressed} cells</strong></div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Scatter
                data={PARETO_SOLUTIONS}
                onClick={(node: any) => setSelectedSolution(node as ParetoSolution)}
              >
                {PARETO_SOLUTIONS.map((entry, index) => (
                  <Cell
                    key={`point-${index}`}
                    fill={selectedSolution.id === entry.id ? '#38bdf8' : '#f97316'}
                    stroke={selectedSolution.id === entry.id ? '#fff' : 'none'}
                    strokeWidth={2}
                    r={selectedSolution.id === entry.id ? 9 : 7}
                    style={{ cursor: 'pointer' }}
                  />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        {/* Selected Solution Details */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <div style={{ background: 'rgba(56, 189, 248, 0.08)', padding: '18px', borderRadius: '12px', border: '1px solid rgba(56, 189, 248, 0.3)' }}>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
              Selected Solution
            </div>
            <div style={{ fontSize: '1.25rem', fontWeight: 800, fontFamily: 'var(--font-display)', color: 'var(--accent-cyan)', marginTop: '2px' }}>
              {selectedSolution.name}
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '6px', lineHeight: 1.45 }}>
              {selectedSolution.strategy}
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
            <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Hotspots Addressed</div>
              <div style={{ fontSize: '1.3rem', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-display)' }}>
                {selectedSolution.hotspots_addressed}{' '}
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>cells</span>
              </div>
            </div>
            <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Equity & Vulnerability Index</div>
              <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#10b981', fontFamily: 'var(--font-display)' }}>
                {(selectedSolution.equity_score * 100).toFixed(0)}%
              </div>
            </div>
          </div>

          <div style={{ background: 'rgba(15, 23, 42, 0.4)', padding: '14px', borderRadius: '8px', border: '1px solid var(--border-subtle)', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            <strong style={{ color: 'var(--accent-orange)' }}>📌 Municipal Implementation Plan:</strong>{' '}
            Deploy in Phase 1 focusing on high-density residential areas in Surat's central & eastern zones.
            Precomputed cooling surrogate achieves <strong>94% convergence speedup</strong> over full simulation during NSGA-II iterations.
          </div>
        </div>
      </div>
    </div>
  );
};
