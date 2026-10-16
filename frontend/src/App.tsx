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
  const [selectedCell, setSelectedCell] = useState<GridCell | null>(null);
  const [showHotspotsOnly, setShowHotspotsOnly] = useState<boolean>(false);

  // Get current season cells
  const cells = SURAT_GRID_CELLS[currentSeason];

  // Derive city-wide KPIs for current season
  const avgTemp = (cells.reduce((acc, c) => acc + c.lst_c, 0) / cells.length).toFixed(1);
  const maxTemp = Math.max(...cells.map((c) => c.lst_c)).toFixed(1);
  const hotspotCount = cells.filter((c) => c.is_hotspot).length;

  const handleSeasonChange = (season: Season) => {
    setCurrentSeason(season);
    // Keep selected cell if exists in new season, otherwise clear
    if (selectedCell) {
      const match = SURAT_GRID_CELLS[season].find((c) => c.cell_id === selectedCell.cell_id);
      setSelectedCell(match || null);
    }
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
      </header>

      {/* Main Workspace */}
      <main style={{ flex: 1, padding: '18px 24px', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* KPI Metrics Strip */}
        <div className="kpi-grid">
          <div className="glass-panel kpi-card alert">
            <span className="kpi-label">Mean City Surface LST</span>
            <span className="kpi-value">{avgTemp}°C</span>
            <span className="kpi-sub">Across 876 Surat analytical cells</span>
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
    </div>
  );
};

export default App;
