import React from 'react';
import { GridCell } from '../types';
import { getCellSHAPDrivers } from '../data/mockData';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts';

interface CellInspectorProps {
  cell: GridCell | null;
}

export const CellInspector: React.FC<CellInspectorProps> = ({ cell }) => {
  if (!cell) {
    return (
      <div className="glass-panel" style={{ padding: '28px', textAlign: 'center', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
        <div style={{ fontSize: '2.5rem', marginBottom: '12px' }}>🎯</div>
        <h3 style={{ fontSize: '1.1rem', marginBottom: '8px', color: 'var(--text-primary)' }}>Select a Grid Cell</h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
          Click any 100m cell on the Surat map to inspect its satellite thermal properties and TreeSHAP explainability attributions.
        </p>
      </div>
    );
  }

  const shapDrivers = getCellSHAPDrivers(cell);

  return (
    <div className="glass-panel" style={{ padding: '20px', height: '100%', display: 'flex', flexDirection: 'column', gap: '16px', overflowY: 'auto' }}>
      {/* Header with Hotspot Badge */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
            Cell #{cell.cell_id} · Surat Municipal Limits
          </div>
          <div style={{ fontSize: '1.5rem', fontWeight: 700, fontFamily: 'var(--font-display)', color: 'var(--text-primary)' }}>
            {cell.lst_c.toFixed(1)}°C{' '}
            <span style={{ fontSize: '0.9rem', color: cell.anomaly_c > 0 ? '#ef4444' : '#10b981', fontWeight: 600 }}>
              ({cell.anomaly_c > 0 ? '+' : ''}{cell.anomaly_c.toFixed(1)}°C anomaly)
            </span>
          </div>
        </div>
        {cell.is_hotspot ? (
          <span style={{ background: 'rgba(239, 68, 68, 0.2)', color: '#ef4444', border: '1px solid rgba(239, 68, 68, 0.4)', padding: '4px 10px', borderRadius: '20px', fontSize: '0.75rem', fontWeight: 700 }}>
            🔥 HOTSPOT
          </span>
        ) : (
          <span style={{ background: 'rgba(16, 185, 129, 0.2)', color: '#10b981', border: '1px solid rgba(16, 185, 129, 0.4)', padding: '4px 10px', borderRadius: '20px', fontSize: '0.75rem', fontWeight: 600 }}>
            NORMAL RANGE
          </span>
        )}
      </div>

      {/* Surface Descriptors Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
        <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '10px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Built-up Area</div>
          <div style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)' }}>{(cell.built_up_fraction * 100).toFixed(0)}%</div>
        </div>
        <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '10px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Vegetation (NDVI)</div>
          <div style={{ fontSize: '1.05rem', fontWeight: 600, color: '#10b981' }}>{cell.ndvi.toFixed(2)}</div>
        </div>
        <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '10px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Albedo (Reflectance)</div>
          <div style={{ fontSize: '1.05rem', fontWeight: 600, color: '#38bdf8' }}>{cell.albedo.toFixed(2)}</div>
        </div>
        <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '10px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Road Density</div>
          <div style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)' }}>{cell.road_density.toFixed(1)} km/km²</div>
        </div>
        <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '10px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Dist to Coast</div>
          <div style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)' }}>{(cell.dist_to_coast_m / 1000).toFixed(1)} km</div>
        </div>
        <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '10px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Elevation</div>
          <div style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)' }}>{cell.elevation_m.toFixed(0)} m</div>
        </div>
      </div>

      {/* TreeSHAP Explainability Waterfall */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
            TreeSHAP Feature Contributions (°C)
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
            Positive = Warming · Negative = Cooling
          </div>
        </div>

        <div style={{ width: '100%', height: '220px', background: 'rgba(15, 23, 42, 0.4)', borderRadius: '8px', padding: '8px', border: '1px solid var(--border-subtle)' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={shapDrivers}
              layout="vertical"
              margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
            >
              <XAxis type="number" stroke="#64748b" fontSize={11} domain={[-3.5, 3.5]} tickFormatter={(v: number | string) => `${Number(v) > 0 ? '+' : ''}${v}°C`} />
              <YAxis
                type="category"
                dataKey="feature"
                stroke="#64748b"
                fontSize={10}
                width={95}
                tickFormatter={(val: string) => {
                  const map: Record<string, string> = {
                    built_up_fraction: 'Built-Up',
                    ndvi: 'NDVI (Greenery)',
                    albedo: 'Albedo',
                    dist_to_coast_m: 'Coast Distance',
                    road_density: 'Road Density',
                    dist_to_water_m: 'Water Distance',
                  };
                  return map[val] || val;
                }}
              />
              <Tooltip
                contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px', fontSize: '12px' }}
                formatter={(val: number) => [`${val > 0 ? '+' : ''}${val.toFixed(2)} °C`, 'SHAP Impact']}
              />
              <ReferenceLine x={0} stroke="#475569" strokeDasharray="3 3" />
              <Bar dataKey="shap_value" radius={[4, 4, 4, 4]}>
                {shapDrivers.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.shap_value > 0 ? '#ef4444' : '#10b981'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* AI Synthesis Summary */}
      <div style={{ background: 'rgba(56, 189, 248, 0.08)', border: '1px solid rgba(56, 189, 248, 0.2)', padding: '12px', borderRadius: '8px', fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
        <strong style={{ color: 'var(--accent-cyan)' }}>🧠 Explainability Insight:</strong>{' '}
        This cell's elevated thermal anomaly (+{cell.anomaly_c.toFixed(1)}°C) is primarily driven by{' '}
        <strong style={{ color: '#fff' }}>high built-up density ({(cell.built_up_fraction * 100).toFixed(0)}%)</strong> and{' '}
        <strong style={{ color: '#fff' }}>low vegetative cover (NDVI: {cell.ndvi.toFixed(2)})</strong>. Adding cool roofs or tree canopy will produce the highest cooling leverage.
      </div>
    </div>
  );
};
