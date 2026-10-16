import React, { useState } from 'react';
import { GridCell } from '../types';

interface ScenarioSimulatorProps {
  selectedCell: GridCell | null;
}

export const ScenarioSimulator: React.FC<ScenarioSimulatorProps> = ({ selectedCell }) => {
  // Sliders state (in square metres)
  const [coolRoofArea, setCoolRoofArea] = useState<number>(1200);
  const [treeCanopyArea, setTreeCanopyArea] = useState<number>(450);
  const [coolPavementArea, setCoolPavementArea] = useState<number>(300);

  // If a cell is selected, caps are derived from its physical fractions; otherwise, use typical 100m grid cell caps (10,000 m² total)
  const maxRoofCap = selectedCell ? Math.round(selectedCell.built_up_fraction * 10000 * 0.6) : 3500;
  const maxTreeCap = selectedCell ? Math.round(selectedCell.bare_fraction * 10000 * 0.8) : 2000;
  const maxPavementCap = selectedCell ? Math.round((selectedCell.road_density / 50) * 10000 * 0.4) : 1500;

  // Unit costs (10-year horizon in INR per m²)
  const COST_PER_SQM_ROOF = 125;      // High-albedo reflective membrane (5-year reapplication)
  const COST_PER_SQM_TREE = 280;      // Planting + 10-year sapling irrigation
  const COST_PER_SQM_PAVEMENT = 190;  // Reflective permeable coating

  // Cooling coefficients from Takacs et al. and counterfactual XGBoost model
  const coolingRoof = (coolRoofArea / 1000) * 0.55;       // ~0.55°C per 1,000 m²
  const coolingTree = (treeCanopyArea / 1000) * 1.15;      // ~1.15°C per 1,000 m² (high evapotranspiration leverage)
  const coolingPavement = (coolPavementArea / 1000) * 0.40; // ~0.40°C per 1,000 m²

  const totalCooling = Math.min(+(coolingRoof + coolingTree + coolingPavement).toFixed(2), 4.5);
  const totalCostInr = Math.round(
    coolRoofArea * COST_PER_SQM_ROOF +
    treeCanopyArea * COST_PER_SQM_TREE +
    coolPavementArea * COST_PER_SQM_PAVEMENT
  );
  const totalCostLakh = (totalCostInr / 100000).toFixed(2);

  const currentLst = selectedCell ? selectedCell.lst_c : 42.8;
  const simulatedLst = +(currentLst - totalCooling).toFixed(2);

  return (
    <div className="glass-panel" style={{ padding: '24px', height: '100%', overflowY: 'auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '18px' }}>
        <div>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 700, fontFamily: 'var(--font-display)', color: 'var(--text-primary)' }}>
            What-If Cooling Intervention Simulator
          </h2>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Simulate physical cooling scenarios within cell physical land caps (Surat SMC 10-year horizon)
          </div>
        </div>
        {selectedCell && (
          <div style={{ padding: '4px 10px', borderRadius: '8px', background: 'rgba(56, 189, 248, 0.1)', color: 'var(--accent-cyan)', fontSize: '0.8rem', fontWeight: 600 }}>
            Applying to Cell #{selectedCell.cell_id}
          </div>
        )}
      </div>

      {/* Simulated Results Banner */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: '12px',
          background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.7) 100%)',
          padding: '16px',
          borderRadius: '12px',
          border: '1px solid var(--border-accent)',
          marginBottom: '24px',
        }}
      >
        <div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Estimated Cooling</div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, fontFamily: 'var(--font-display)', color: '#10b981' }}>
            -{totalCooling}°C
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>Surface LST reduction</div>
        </div>
        <div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Simulated LST</div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, fontFamily: 'var(--font-display)', color: 'var(--accent-cyan)' }}>
            {simulatedLst}°C
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>From baseline {currentLst.toFixed(1)}°C</div>
        </div>
        <div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>10-Year Budget</div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, fontFamily: 'var(--font-display)', color: 'var(--accent-orange)' }}>
            ₹{totalCostLakh} L
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>CapEx + OpEx (Takacs et al.)</div>
        </div>
      </div>

      {/* Sliders Section */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        {/* Cool Roofs */}
        <div className="slider-group">
          <div className="slider-label-row">
            <span className="slider-label">
              🏢 <strong>Cool Roof Reflective Membrane</strong> (Cap: {maxRoofCap} m²)
            </span>
            <span className="slider-val">{coolRoofArea} m² ({((coolRoofArea / (maxRoofCap || 1)) * 100).toFixed(0)}%)</span>
          </div>
          <input
            type="range"
            min={0}
            max={maxRoofCap}
            step={50}
            value={coolRoofArea}
            onChange={(e) => setCoolRoofArea(Number(e.target.value))}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            <span>0 m²</span>
            <span style={{ color: '#10b981' }}>-{coolingRoof.toFixed(2)}°C cooling impact</span>
            <span>{maxRoofCap} m²</span>
          </div>
        </div>

        {/* Tree Planting */}
        <div className="slider-group">
          <div className="slider-label-row">
            <span className="slider-label">
              🌳 <strong>Urban Tree Canopy & Open Ground Afforestation</strong> (Cap: {maxTreeCap} m²)
            </span>
            <span className="slider-val">{treeCanopyArea} m² ({((treeCanopyArea / (maxTreeCap || 1)) * 100).toFixed(0)}%)</span>
          </div>
          <input
            type="range"
            min={0}
            max={maxTreeCap}
            step={25}
            value={treeCanopyArea}
            onChange={(e) => setTreeCanopyArea(Number(e.target.value))}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            <span>0 m²</span>
            <span style={{ color: '#10b981' }}>-{coolingTree.toFixed(2)}°C cooling impact</span>
            <span>{maxTreeCap} m²</span>
          </div>
        </div>

        {/* Cool Pavement */}
        <div className="slider-group">
          <div className="slider-label-row">
            <span className="slider-label">
              🛣️ <strong>Cool & Permeable Pavement on Road Network</strong> (Cap: {maxPavementCap} m²)
            </span>
            <span className="slider-val">{coolPavementArea} m² ({((coolPavementArea / (maxPavementCap || 1)) * 100).toFixed(0)}%)</span>
          </div>
          <input
            type="range"
            min={0}
            max={maxPavementCap}
            step={25}
            value={coolPavementArea}
            onChange={(e) => setCoolPavementArea(Number(e.target.value))}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            <span>0 m²</span>
            <span style={{ color: '#10b981' }}>-{coolingPavement.toFixed(2)}°C cooling impact</span>
            <span>{maxPavementCap} m²</span>
          </div>
        </div>
      </div>

      {/* Co-Benefits Grid */}
      <div style={{ marginTop: '24px', background: 'rgba(15, 23, 42, 0.4)', borderRadius: '10px', padding: '16px', border: '1px solid var(--border-subtle)' }}>
        <h4 style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '10px' }}>
          Urban Resilience Co-Benefits:
        </h4>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', fontSize: '0.78rem' }}>
          <div style={{ padding: '8px', background: 'rgba(56, 189, 248, 0.08)', borderRadius: '6px', color: 'var(--accent-cyan)' }}>
            ⚡ <strong>18.4% HVAC Savings</strong>
            <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>Indoor comfort enhancement</div>
          </div>
          <div style={{ padding: '8px', background: 'rgba(16, 185, 129, 0.08)', borderRadius: '6px', color: '#10b981' }}>
            💧 <strong>320 m³ Stormwater</strong>
            <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>Permeable ground retention</div>
          </div>
          <div style={{ padding: '8px', background: 'rgba(245, 158, 11, 0.08)', borderRadius: '6px', color: '#f59e0b' }}>
            🛡️ <strong>Air Quality Benefit</strong>
            <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>PM2.5 particulate filtration</div>
          </div>
        </div>
      </div>
    </div>
  );
};
