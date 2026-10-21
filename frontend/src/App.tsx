import React, { useState } from 'react';
import { Season, GridCell } from './types';
import { SURAT_GRID_CELLS } from './data/mockData';
import { MapView } from './components/MapView';
import { CellInspector } from './components/CellInspector';
import { ScenarioSimulator } from './components/ScenarioSimulator';
import { ParetoExplorer } from './components/ParetoExplorer';
import { BenchmarkView } from './components/BenchmarkView';

export const App: React.FC = () => {
  const [currentSeason, setCurrentSeason] = useState<Season>('pre_monsoon');
  const [activeTab, setActiveTab] = useState<'map' | 'simulator' | 'pareto' | 'benchmarks'>('map');
  const [selectedCell, setSelectedCell] = useState<GridCell | null>(() => {
    return SURAT_GRID_CELLS['pre_monsoon'].find((c) => c.is_hotspot) || SURAT_GRID_CELLS['pre_monsoon'][0] || null;
  });
  const [showHotspotsOnly, setShowHotspotsOnly] = useState<boolean>(false);

  // Get current season cells
  const cells = SURAT_GRID_CELLS[currentSeason];

  // Derive city-wide KPIs for current season
  const avgTemp = (cells.reduce((acc, c) => acc + c.lst_c, 0) / cells.length).toFixed(1);
  const maxTemp = Math.max(...cells.map((c) => c.lst_c)).toFixed(1);
  const hotspotCount = cells.filter((c) => c.is_hotspot).length;

  const handleSeasonChange = (season: Season) => {
    setCurrentSeason(season);
    // Keep selected cell if exists in new season, otherwise select first hotspot
    if (selectedCell) {
      const match = SURAT_GRID_CELLS[season].find((c) => c.cell_id === selectedCell.cell_id);
      setSelectedCell(match || SURAT_GRID_CELLS[season].find((c) => c.is_hotspot) || null);
    }
  };

  const [showBriefModal, setShowBriefModal] = useState<boolean>(false);

  const handleExportReport = () => {
    const reportData = {
      project: 'HeatSense AI — Surat Urban Heat Mitigation & Decision Support System',
      target_city: 'Surat, Gujarat, India',
      season: currentSeason,
      generated_at: new Date().toISOString(),
      grid_resolution: '100m x 100m (868 analytical cells)',
      city_summary: {
        mean_lst_c: avgTemp,
        peak_lst_c: maxTemp,
        total_hotspots_count: hotspotCount,
        max_cooling_feasibility_c: -4.5,
      },
      top_hotspots: cells
        .filter((c) => c.is_hotspot)
        .sort((a, b) => b.lst_c - a.lst_c)
        .slice(0, 20)
        .map((c) => ({
          cell_id: c.cell_id,
          area_name: c.area_name,
          zone_name: c.zone_name,
          observed_lst_c: c.lst_c,
          thermal_anomaly_c: c.anomaly_c,
          built_up_fraction: c.built_up_fraction,
          ndvi: c.ndvi,
          primary_driver: c.built_up_fraction > 0.7 ? 'High Impervious Surface' : 'Low Vegetation Cover',
          recommended_intervention: c.built_up_fraction > 0.7 ? 'High-Albedo Cool Roof Coating' : 'Urban Tree Canopy Planting',
        })),
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `HeatSense_Surat_Assessment_${currentSeason}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw', background: 'var(--bg-main)' }}>
      {/* Top Application Header */}
      <header className="app-header">
        <div className="brand-badge">
          <div style={{ fontSize: '1.6rem' }}>🛰️</div>
          <div>
            <div className="brand-title">HeatSense AI</div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              Urban Heat Mitigation & Decision Support System
            </div>
          </div>
          <span className="brand-tag">Surat · 100m Grid</span>
        </div>

        {/* View Navigation Tabs */}
        <nav className="nav-tabs">
          <button
            className={`nav-tab-btn ${activeTab === 'map' ? 'active' : ''}`}
            onClick={() => setActiveTab('map')}
          >
            🗺️ HeatMap & SHAP
          </button>
          <button
            className={`nav-tab-btn ${activeTab === 'simulator' ? 'active' : ''}`}
            onClick={() => setActiveTab('simulator')}
          >
            💡 What-If Simulator
          </button>
          <button
            className={`nav-tab-btn ${activeTab === 'pareto' ? 'active' : ''}`}
            onClick={() => setActiveTab('pareto')}
          >
            ⚖️ NSGA-II Pareto Front
          </button>
          <button
            className={`nav-tab-btn ${activeTab === 'benchmarks' ? 'active' : ''}`}
            onClick={() => setActiveTab('benchmarks')}
          >
            📊 Model Benchmarks
          </button>
        </nav>

        {/* Season Selector */}
        <div className="season-selector">
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginRight: '4px' }}>
            Season:
          </span>
          <button
            className={`season-btn ${currentSeason === 'pre_monsoon' ? 'active' : ''}`}
            onClick={() => handleSeasonChange('pre_monsoon')}
            title="Summer / Peak Heat Wave Season (Mar–May)"
          >
            ☀️ Summer (Pre-Monsoon)
          </button>
          <button
            className={`season-btn ${currentSeason === 'monsoon' ? 'active' : ''}`}
            onClick={() => handleSeasonChange('monsoon')}
            title="Monsoon Season (Jun–Sep)"
          >
            🌧️ Monsoon
          </button>
          <button
            className={`season-btn ${currentSeason === 'post_monsoon' ? 'active' : ''}`}
            onClick={() => handleSeasonChange('post_monsoon')}
            title="Post-Monsoon (Oct–Nov)"
          >
            🍂 Post-Monsoon
          </button>
          <button
            className={`season-btn ${currentSeason === 'winter' ? 'active' : ''}`}
            onClick={() => handleSeasonChange('winter')}
            title="Winter (Dec–Feb)"
          >
            ❄️ Winter
          </button>
        </div>

        {/* Action Controls: Policy Brief & Export Report */}
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <button
            onClick={() => setShowBriefModal(true)}
            style={{
              background: 'rgba(99, 102, 241, 0.18)',
              color: '#c7d2fe',
              border: '1px solid rgba(99, 102, 241, 0.4)',
              padding: '6px 12px',
              borderRadius: '8px',
              fontSize: '0.78rem',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              transition: 'all 0.2s ease',
            }}
            title="View Surat Municipal Action Plan & Policy Brief"
          >
            <span>📋</span>
            <span>Policy Brief</span>
          </button>
          <button
            onClick={handleExportReport}
            style={{
              background: 'rgba(16, 185, 129, 0.18)',
              color: '#6ee7b7',
              border: '1px solid rgba(16, 185, 129, 0.4)',
              padding: '6px 12px',
              borderRadius: '8px',
              fontSize: '0.78rem',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              transition: 'all 0.2s ease',
            }}
            title="Download complete assessment report as JSON"
          >
            <span>📥</span>
            <span>Export Assessment</span>
          </button>
        </div>
      </header>

      {/* Main Workspace */}
      <main style={{ flex: 1, padding: '18px 24px', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* KPI Metrics Strip */}
        <div className="kpi-grid">
          <div className="glass-panel kpi-card alert">
            <span className="kpi-label">Mean City Surface LST</span>
            <span className="kpi-value">{avgTemp}°C</span>
            <span className="kpi-sub">Across {cells.length} Surat analytical cells</span>
          </div>
          <div className="glass-panel kpi-card hot">
            <span className="kpi-label">Peak Hotspot LST</span>
            <span className="kpi-value" style={{ color: '#ef4444' }}>{maxTemp}°C</span>
            <span className="kpi-sub">Max thermal anomaly cell</span>
          </div>
          <div className="glass-panel kpi-card alert">
            <span className="kpi-label">Identified Hotspots</span>
            <span className="kpi-value" style={{ color: '#f59e0b' }}>{hotspotCount} Cells</span>
            <span className="kpi-sub">High vulnerability 100m zones</span>
          </div>
          <div className="glass-panel kpi-card optimal">
            <span className="kpi-label">Max Cooling Feasibility</span>
            <span className="kpi-value" style={{ color: '#10b981' }}>-4.5°C</span>
            <span className="kpi-sub">Via combined green/cool measures</span>
          </div>
        </div>

        {/* Dynamic Content Views */}
        <div style={{ flex: 1, minHeight: 0 }}>
          {activeTab === 'map' && (
            <div style={{ display: 'grid', gridTemplateColumns: '1.35fr 0.85fr', gap: '18px', height: '100%' }}>
              {/* Left: Map with Hotspot Filter */}
              <div style={{ position: 'relative', height: '100%' }}>
                <div style={{ position: 'absolute', top: '14px', right: '14px', zIndex: 999 }}>
                  <button
                    onClick={() => setShowHotspotsOnly(!showHotspotsOnly)}
                    style={{
                      background: showHotspotsOnly ? 'var(--accent-rose)' : 'rgba(15, 23, 42, 0.85)',
                      color: '#fff',
                      border: '1px solid var(--border-subtle)',
                      padding: '8px 14px',
                      borderRadius: '8px',
                      fontSize: '0.8rem',
                      fontWeight: 600,
                      cursor: 'pointer',
                      backdropFilter: 'blur(8px)',
                      boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
                    }}
                  >
                    {showHotspotsOnly ? '🔥 Showing Hotspots Only' : '🗺️ Show All Grid Cells'}
                  </button>
                </div>
                <MapView
                  cells={cells}
                  selectedCell={selectedCell}
                  onSelectCell={(c) => setSelectedCell(c)}
                  showHotspotsOnly={showHotspotsOnly}
                />
              </div>

              {/* Right: Cell Inspector & SHAP Explainability */}
              <div style={{ height: '100%' }}>
                <CellInspector cell={selectedCell} />
              </div>
            </div>
          )}

          {activeTab === 'simulator' && (
            <ScenarioSimulator selectedCell={selectedCell} />
          )}

          {activeTab === 'pareto' && (
            <ParetoExplorer />
          )}

          {activeTab === 'benchmarks' && (
            <BenchmarkView currentSeason={currentSeason} />
          )}
        </div>
      </main>

      {/* Municipal Policy Brief & Executive Summary Modal */}
      {showBriefModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0, 0, 0, 0.75)',
            backdropFilter: 'blur(8px)',
            zIndex: 9999,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '24px',
          }}
          onClick={() => setShowBriefModal(false)}
        >
          <div
            className="glass-panel"
            style={{
              width: '100%',
              maxWidth: '740px',
              maxHeight: '85vh',
              overflowY: 'auto',
              padding: '28px',
              background: 'rgba(15, 23, 42, 0.95)',
              border: '1px solid rgba(56, 189, 248, 0.3)',
              boxShadow: '0 20px 50px rgba(0,0,0,0.8)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
              <div>
                <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--accent-cyan)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Surat Municipal Corporation (SMC) · Decision Support
                </div>
                <h2 style={{ fontSize: '1.4rem', fontWeight: 700, fontFamily: 'var(--font-display)', color: 'var(--text-primary)', marginTop: '2px' }}>
                  Surat Urban Heat Action & Mitigation Brief
                </h2>
              </div>
              <button
                onClick={() => setShowBriefModal(false)}
                style={{
                  background: 'rgba(255,255,255,0.1)',
                  border: 'none',
                  color: 'var(--text-muted)',
                  fontSize: '1.2rem',
                  padding: '4px 10px',
                  borderRadius: '6px',
                  cursor: 'pointer',
                }}
              >
                ✕
              </button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px', marginBottom: '20px' }}>
              <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', padding: '10px', borderRadius: '8px' }}>
                <div style={{ fontSize: '0.7rem', color: '#fca5a5' }}>Peak Hotspot LST</div>
                <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#ef4444' }}>{maxTemp}°C</div>
                <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>Pandesara / Varachha</div>
              </div>
              <div style={{ background: 'rgba(245, 158, 11, 0.1)', border: '1px solid rgba(245, 158, 11, 0.3)', padding: '10px', borderRadius: '8px' }}>
                <div style={{ fontSize: '0.7rem', color: '#fde68a' }}>Critical Hotspots</div>
                <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#f59e0b' }}>{hotspotCount} Cells</div>
                <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>100m analytical zones</div>
              </div>
              <div style={{ background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.3)', padding: '10px', borderRadius: '8px' }}>
                <div style={{ fontSize: '0.7rem', color: '#a7f3d0' }}>Max Cooling Target</div>
                <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#10b981' }}>-4.5°C</div>
                <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>Combined Green-Cool</div>
              </div>
              <div style={{ background: 'rgba(99, 102, 241, 0.1)', border: '1px solid rgba(99, 102, 241, 0.3)', padding: '10px', borderRadius: '8px' }}>
                <div style={{ fontSize: '0.7rem', color: '#c7d2fe' }}>Pareto Optimal Budget</div>
                <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#818cf8' }}>₹54.0 L</div>
                <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>Balanced Plan (310 Cells)</div>
              </div>
            </div>

            <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '8px' }}>
              🎯 Top 4 Priority Intervention Wards
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '20px' }}>
              <div style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid var(--border-subtle)', padding: '10px 14px', borderRadius: '8px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <strong style={{ color: '#ef4444' }}>1. Pandesara GIDC Industrial Cluster (South Zone)</strong>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Primary driver: 95% built-up impervious concrete & chemical/dyeing industrial sheds.</div>
                </div>
                <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#ef4444' }}>45.8°C (+6.5°C)</span>
              </div>
              <div style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid var(--border-subtle)', padding: '10px 14px', borderRadius: '8px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <strong style={{ color: '#f97316' }}>2. Varachha Diamond Hub (East Zone)</strong>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Primary driver: Extreme building density, narrow street canyons & zero vegetative cover (NDVI 0.06).</div>
                </div>
                <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#f97316' }}>45.8°C (+6.5°C)</span>
              </div>
              <div style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid var(--border-subtle)', padding: '10px 14px', borderRadius: '8px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <strong style={{ color: '#f59e0b' }}>3. Katargam Core & Ved Road Craft Hub (North Zone)</strong>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Primary driver: Diamond cutting workshops with high thermal emissivity and unshaded rooftops.</div>
                </div>
                <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#f59e0b' }}>45.3°C (+6.0°C)</span>
              </div>
              <div style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid var(--border-subtle)', padding: '10px 14px', borderRadius: '8px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <strong style={{ color: '#f59e0b' }}>4. Begampura & Central Ring Road (Central Zone)</strong>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Primary driver: Wholesale textile market traffic density & low surface reflectance (Albedo 0.09).</div>
                </div>
                <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#f59e0b' }}>44.7°C (+5.4°C)</span>
              </div>
            </div>

            <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '8px' }}>
              🌱 Recommended 3-Tier Action Plan
            </h3>
            <ul style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.6, paddingLeft: '20px', marginBottom: '24px' }}>
              <li><strong>Tier 1 — High-Albedo Cool Roof Mandate:</strong> Subsidized lime-wash and reflective membranes on 2,500 industrial and tin-roof residential units in Pandesara & Varachha (Cost: ₹125/m², Cooling: ~1.5°C).</li>
              <li><strong>Tier 2 — Urban Tree Canopy Afforestation:</strong> Native fast-growing shade species (Neem, Gulmohar, Peepal) planted along Udhna-Navsari highway, Ring Road, and canal corridors (Cooling: ~2.3°C).</li>
              <li><strong>Tier 3 — Permeable Cool Pavements:</strong> High-reflectance porous paving blocks in parking zones and textile transit nodes to reduce ground solar absorption.</li>
            </ul>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
              <button
                onClick={handleExportReport}
                style={{
                  background: 'var(--accent-emerald)',
                  color: '#070b14',
                  border: 'none',
                  padding: '8px 16px',
                  borderRadius: '6px',
                  fontSize: '0.82rem',
                  fontWeight: 700,
                  cursor: 'pointer',
                }}
              >
                📥 Download Full Assessment (.json)
              </button>
              <button
                onClick={() => setShowBriefModal(false)}
                style={{
                  background: 'rgba(255,255,255,0.1)',
                  color: '#fff',
                  border: '1px solid var(--border-subtle)',
                  padding: '8px 16px',
                  borderRadius: '6px',
                  fontSize: '0.82rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
