import React from 'react';
import { MODEL_BENCHMARKS } from '../data/mockData';
import { Season } from '../types';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
} from 'recharts';

interface BenchmarkViewProps {
  currentSeason: Season;
}

export const BenchmarkView: React.FC<BenchmarkViewProps> = ({ currentSeason }) => {
  const filteredBenchmarks = MODEL_BENCHMARKS.filter((b) => b.season === currentSeason);

  const chartData = filteredBenchmarks.map((b) => ({
    name: b.model,
    val_mae: b.val_mae,
    val_rmse: b.val_rmse,
    train_time_s: b.train_time_s,
  }));

  const seasonLabels: Record<Season, string> = {
    pre_monsoon: 'Pre-Monsoon (Summer / Peak Heat)',
    monsoon: 'Monsoon (Rainy Season)',
    post_monsoon: 'Post-Monsoon (Autumn / Clear Sky)',
    winter: 'Winter (Dry Cool Season)',
  };

  return (
    <div className="glass-panel" style={{ padding: '24px', height: '100%', overflowY: 'auto' }}>
      <div style={{ marginBottom: '18px' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 700, fontFamily: 'var(--font-display)', color: 'var(--text-primary)' }}>
          Model Benchmarks & Geospatial Foundation Models
        </h2>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          Validation metrics for {seasonLabels[currentSeason]} across XGBoost baseline and GPU-fine-tuned foundation architectures
        </div>
      </div>

      {/* Comparison Chart */}
      <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '16px', borderRadius: '12px', border: '1px solid var(--border-subtle)', height: '280px', marginBottom: '20px' }}>
        <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px' }}>
          Error Comparison (Lower is Better - °C)
        </div>
        <ResponsiveContainer width="100%" height="85%">
          <BarChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} />
            <XAxis dataKey="name" stroke="#64748b" fontSize={12} />
            <YAxis stroke="#64748b" fontSize={11} domain={[0, 5]} unit="°C" />
            <Tooltip
              contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px', fontSize: '12px' }}
              formatter={(val: number) => [`${val} °C`]}
            />
            <Legend wrapperStyle={{ fontSize: '12px' }} />
            <Bar dataKey="val_mae" name="Validation MAE (°C)" fill="#38bdf8" radius={[4, 4, 0, 0]} />
            <Bar dataKey="val_rmse" name="Validation RMSE (°C)" fill="#f97316" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Benchmark Table */}
      <div style={{ overflowX: 'auto', marginBottom: '20px' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border-subtle)', textAlign: 'left', color: 'var(--text-muted)' }}>
              <th style={{ padding: '10px' }}>Model</th>
              <th style={{ padding: '10px' }}>Architecture Type</th>
              <th style={{ padding: '10px' }}>Val MAE (°C)</th>
              <th style={{ padding: '10px' }}>Val RMSE (°C)</th>
              <th style={{ padding: '10px' }}>Val R²</th>
              <th style={{ padding: '10px' }}>Best Epoch/Iter</th>
              <th style={{ padding: '10px' }}>GPU Time</th>
            </tr>
          </thead>
          <tbody>
            {filteredBenchmarks.map((b, idx) => (
              <tr key={idx} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                <td style={{ padding: '10px', fontWeight: 600, color: 'var(--text-primary)' }}>
                  {b.model}
                </td>
                <td style={{ padding: '10px', color: 'var(--text-secondary)' }}>{b.model_type}</td>
                <td style={{ padding: '10px', fontWeight: 700, color: b.val_mae < 1.0 ? '#10b981' : 'var(--text-primary)' }}>
                  {b.val_mae.toFixed(3)}
                </td>
                <td style={{ padding: '10px', color: 'var(--text-secondary)' }}>{b.val_rmse.toFixed(3)}</td>
                <td style={{ padding: '10px', color: b.val_r2 > 0 ? '#10b981' : '#f43f5e' }}>{b.val_r2.toFixed(3)}</td>
                <td style={{ padding: '10px', color: 'var(--text-muted)' }}>{b.best_epoch}</td>
                <td style={{ padding: '10px', color: 'var(--text-muted)' }}>{b.train_time_s.toFixed(2)}s</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Model Takeaways */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
        <div style={{ background: 'rgba(56, 189, 248, 0.08)', padding: '14px', borderRadius: '10px', border: '1px solid rgba(56, 189, 248, 0.25)', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
          <strong style={{ color: 'var(--accent-cyan)' }}>NASA-IBM Prithvi-EO 2.0:</strong> Best performing foundation ViT with spatio-spectral tokenization, learning non-linear cross-band interactions between NDVI, albedo, and surface moisture.
        </div>
        <div style={{ background: 'rgba(249, 115, 22, 0.08)', padding: '14px', borderRadius: '10px', border: '1px solid rgba(249, 115, 22, 0.25)', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
          <strong style={{ color: 'var(--accent-orange)' }}>XGBoost Baseline:</strong> Exceptional tabular fitting with fast convergence (&lt;0.6s) and stable 5-fold spatial block cross-validation ensuring zero autocorrelation leakage.
        </div>
      </div>
    </div>
  );
};
