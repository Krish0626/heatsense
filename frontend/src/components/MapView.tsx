import React, { useEffect, useMemo, useRef, useState } from 'react';
import L from 'leaflet';
import { GridCell } from '../types';
import { ThermalHeatLayer } from '../utils/leafletHeat';

interface MapViewProps {
  cells: GridCell[];
  selectedCell: GridCell | null;
  onSelectCell: (cell: GridCell) => void;
  showHotspotsOnly: boolean;
}

// 100% Free Tile Providers — No API Key Required!
const TILE_PROVIDERS = {
  satellite: {
    name: '🛰️ Satellite (Esri)',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attribution: 'Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics',
    maxZoom: 19,
    className: '',
  },
  osm: {
    name: '🗺️ Street Map (OSM)',
    url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19,
    className: '',
  },
  dark: {
    name: '🌙 Dark Mode',
    url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 19,
    className: 'dark-tiles',
  },
};

type TileKey = keyof typeof TILE_PROVIDERS;
type LayerMode = 'combined' | 'heatmap' | 'grid';

// Surat Major Areas & Hotspot Anchor Points
const SURAT_AREA_DEFINITIONS = [
  { id: 'pandesara', name: 'Pandesara GIDC', zone: 'South Zone (Udhna)', lat: 21.140, lon: 72.835, priority: 1 },
  { id: 'varachha', name: 'Varachha Diamond Hub', zone: 'East Zone (Varachha)', lat: 21.220, lon: 72.860, priority: 1 },
  { id: 'amroli', name: 'Amroli & Kosad', zone: 'North Zone (Katargam)', lat: 21.265, lon: 72.835, priority: 1 },
  { id: 'katargam', name: 'Katargam Core', zone: 'North Zone (Katargam)', lat: 21.230, lon: 72.825, priority: 1 },
  { id: 'begampura', name: 'Begampura Textile Hub', zone: 'Central Zone', lat: 21.190, lon: 72.845, priority: 1 },
  { id: 'citycore', name: 'Surat City Core / Ring Rd', zone: 'Central Zone', lat: 21.195, lon: 72.830, priority: 1 },
  { id: 'sachin', name: 'Sachin Heavy Industrial', zone: 'Southern Industrial Belt', lat: 21.075, lon: 72.875, priority: 1 },
  { id: 'bhestan', name: 'Bhestan & Bamroli', zone: 'South Zone (Udhna)', lat: 21.125, lon: 72.860, priority: 1 },
  { id: 'singanpore', name: 'Singanpore & Ved Rd', zone: 'North Zone (Katargam)', lat: 21.220, lon: 72.815, priority: 2 },
  { id: 'limbayat', name: 'Limbayat Settlement', zone: 'South-East Zone', lat: 21.180, lon: 72.880, priority: 2 },
  { id: 'station', name: 'Surat Central Station', zone: 'Central Zone', lat: 21.205, lon: 72.840, priority: 2 },
  { id: 'udhna', name: 'Udhna Industrial Junction', zone: 'South Zone (Udhna)', lat: 21.165, lon: 72.840, priority: 2 },
  { id: 'athwa', name: 'Athwa Lines & Riverfront', zone: 'South-West Zone', lat: 21.175, lon: 72.805, priority: 2 },
  { id: 'adajan', name: 'Adajan Residential', zone: 'West Zone (Rander)', lat: 21.195, lon: 72.790, priority: 2 },
  { id: 'rander', name: 'Rander Historic Town', zone: 'West Zone (Rander)', lat: 21.215, lon: 72.795, priority: 2 },
  { id: 'vesu', name: 'Vesu - VIP Road Corridor', zone: 'South-West Zone', lat: 21.140, lon: 72.785, priority: 3 },
  { id: 'piplod', name: 'Piplod - Gaurav Path', zone: 'South-West Zone', lat: 21.150, lon: 72.775, priority: 3 },
  { id: 'sarthana', name: 'Sarthana Nature Park Buffer', zone: 'East Zone', lat: 21.235, lon: 72.910, priority: 3 },
  { id: 'hazira', name: 'Hazira Heavy Port Terminal', zone: 'Western Industrial', lat: 21.120, lon: 72.670, priority: 2 },
  { id: 'dumas', name: 'Dumas Coastal Marine Zone', zone: 'South-West Zone', lat: 21.078, lon: 72.715, priority: 3 },
];

// Color mapping for LST in °C
function getTemperatureColor(temp: number): string {
  if (temp >= 42) return '#ef4444'; // Extreme heat (red)
  if (temp >= 38) return '#f97316'; // High heat (orange)
  if (temp >= 34) return '#f59e0b'; // Moderate heat (amber)
  if (temp >= 30) return '#10b981'; // Mild (emerald)
  return '#3b82f6';                 // Cool (blue)
}

export const MapView: React.FC<MapViewProps> = ({
  cells,
  selectedCell,
  onSelectCell,
  showHotspotsOnly,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const heatLayerRef = useRef<ThermalHeatLayer | null>(null);
  const gridLayerGroupRef = useRef<L.LayerGroup | null>(null);
  const labelLayerGroupRef = useRef<L.LayerGroup | null>(null);
  const tileLayerRef = useRef<L.TileLayer | null>(null);

  const [activeTile, setActiveTile] = useState<TileKey>('satellite');
  const [layerMode, setLayerMode] = useState<LayerMode>('combined');
  const [showAreaLabels, setShowAreaLabels] = useState<boolean>(true);

  // Compute live Area stats (Peak LST & Hotspot status per area) from current season cells
  const areaStats = useMemo(() => {
    return SURAT_AREA_DEFINITIONS.map((def) => {
      // Find cells belonging or close to this anchor
      const nearbyCells = cells.filter((c) => {
        const d = Math.hypot(c.lat - def.lat, (c.lon - def.lon) * 0.93);
        return d < 0.024;
      });

      const effectiveCells = nearbyCells.length > 0 ? nearbyCells : cells;
      const peakCell = effectiveCells.reduce((max, c) => (c.lst_c > max.lst_c ? c : max), effectiveCells[0]);
      const isHotspot = effectiveCells.some((c) => c.is_hotspot) || peakCell.lst_c >= 42.0;

      return {
        ...def,
        peakLst: peakCell.lst_c,
        isHotspot,
        peakCell,
      };
    });
  }, [cells]);

  // Top hotspots list for the quick access bar
  const topHotspots = useMemo(() => {
    return areaStats
      .filter((a) => a.isHotspot)
      .sort((a, b) => b.peakLst - a.peakLst)
      .slice(0, 8);
  }, [areaStats]);

  // 1. Initialize Map instance once
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    // Center on Surat City covering full municipal boundaries
    const map = L.map(mapContainerRef.current, {
      center: [21.175, 72.835],
      zoom: 11,
      zoomControl: true,
    });

    // Basemap Layer
    const provider = TILE_PROVIDERS['satellite'];
    const tileLayer = L.tileLayer(provider.url, {
      attribution: provider.attribution,
      maxZoom: provider.maxZoom,
      className: provider.className,
    }).addTo(map);

    tileLayerRef.current = tileLayer;

    // Grid layer group (polygons)
    const gridGroup = L.layerGroup().addTo(map);
    gridLayerGroupRef.current = gridGroup;

    // Label layer group (Area Markers)
    const labelGroup = L.layerGroup().addTo(map);
    labelLayerGroupRef.current = labelGroup;

    mapInstanceRef.current = map;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // 2. Basemap switcher
  const handleTileChange = (key: TileKey) => {
    if (!mapInstanceRef.current) return;
    setActiveTile(key);

    if (tileLayerRef.current) {
      mapInstanceRef.current.removeLayer(tileLayerRef.current);
    }

    const provider = TILE_PROVIDERS[key];
    const newTileLayer = L.tileLayer(provider.url, {
      attribution: provider.attribution,
      maxZoom: provider.maxZoom,
      className: provider.className,
    }).addTo(mapInstanceRef.current);

    // Keep grid and labels on top
    if (gridLayerGroupRef.current) {
      gridLayerGroupRef.current.eachLayer((layer: any) => {
        if (typeof layer.bringToFront === 'function') layer.bringToFront();
      });
    }

    tileLayerRef.current = newTileLayer;
  };

  // 3. Fit full Surat boundary
  const handleFitSurat = () => {
    if (!mapInstanceRef.current) return;
    mapInstanceRef.current.fitBounds([
      [21.053, 72.704],
      [21.296, 72.974],
    ], { padding: [24, 24] });
  };

  // 4. Fly to specific area
  const handleFlyToArea = (area: typeof areaStats[0]) => {
    if (!mapInstanceRef.current) return;
    mapInstanceRef.current.flyTo([area.lat, area.lon], 13.5, { duration: 1.2 });
    onSelectCell(area.peakCell);
  };

  // 5. Update Smooth Thermal Heatmap Layer (Reference image style)
  useEffect(() => {
    if (!mapInstanceRef.current) return;
    const map = mapInstanceRef.current;

    // Remove existing heat layer if switching away
    if (heatLayerRef.current) {
      map.removeLayer(heatLayerRef.current);
      heatLayerRef.current = null;
    }

    if (layerMode === 'heatmap' || layerMode === 'combined') {
      const displayCells = showHotspotsOnly ? cells.filter((c) => c.is_hotspot) : cells;

      // Format heat points: [lat, lon, normalized_intensity]
      const heatPoints: Array<[number, number, number]> = displayCells.map((c) => {
        // Temperature range normalized 26°C -> 46°C
        const intensity = Math.max(0.12, Math.min(1.0, (c.lst_c - 26) / 19));
        return [c.lat, c.lon, intensity];
      });

      const heatLayer = new ThermalHeatLayer(heatPoints, {
        radius: 36,
        blur: 26,
        minOpacity: layerMode === 'combined' ? 0.45 : 0.65,
        maxZoom: 16,
        max: 1.0,
        gradient: {
          0.12: '#3b82f6', // Cool Blue
          0.30: '#06b6d4', // Cyan
          0.50: '#10b981', // Green
          0.68: '#facc15', // Yellow
          0.82: '#f97316', // Orange
          1.00: '#ef4444', // Glowing Hotspot Red (Matches reference image)
        },
      });

      heatLayer.addTo(map);
      heatLayerRef.current = heatLayer;
    }
  }, [cells, layerMode, showHotspotsOnly]);

  // 6. Update Grid Polygons & Interactive Hover/Touch Tooltips
  useEffect(() => {
    if (!mapInstanceRef.current || !gridLayerGroupRef.current) return;
    const gridGroup = gridLayerGroupRef.current;
    gridGroup.clearLayers();

    const displayCells = showHotspotsOnly ? cells.filter((c) => c.is_hotspot) : cells;

    displayCells.forEach((cell) => {
      const color = getTemperatureColor(cell.lst_c);
      const isSelected = selectedCell?.cell_id === cell.cell_id;
      const latHalf = 0.0042;
      const lonHalf = 0.0043;

      const bounds: L.LatLngBoundsExpression = [
        [cell.lat - latHalf, cell.lon - lonHalf],
        [cell.lat + latHalf, cell.lon + lonHalf],
      ];

      // Opacities change based on layerMode
      let fillOpacity = 0.68;
      let strokeOpacity = 0.4;
      let weight = 0.6;

      if (layerMode === 'heatmap') {
        fillOpacity = isSelected ? 0.35 : 0.001; // transparent hitboxes for heatmap mode
        strokeOpacity = isSelected ? 1 : 0;
        weight = isSelected ? 3.5 : 0;
      } else if (layerMode === 'combined') {
        fillOpacity = isSelected ? 0.5 : 0.18;
        strokeOpacity = isSelected ? 1 : 0.3;
        weight = isSelected ? 3.5 : (cell.is_hotspot ? 1.4 : 0.4);
      } else {
        // Grid mode
        fillOpacity = isSelected ? 0.95 : (cell.is_hotspot ? 0.82 : 0.68);
        strokeOpacity = isSelected ? 1 : (cell.is_hotspot ? 0.9 : 0.3);
        weight = isSelected ? 3.5 : (cell.is_hotspot ? 1.6 : 0.6);
      }

      const rect = L.rectangle(bounds, {
        color: isSelected ? '#38bdf8' : (cell.is_hotspot ? '#ef4444' : 'rgba(15, 23, 42, 0.4)'),
        weight,
        fillColor: color,
        fillOpacity,
        opacity: strokeOpacity,
      });

      if (isSelected) {
        setTimeout(() => {
          try {
            rect.bringToFront();
          } catch {
            // ignore
          }
        }, 10);
      }

      // Rich Area-Aware Tooltip when touching or hovering
      const areaLabel = cell.area_name || 'Surat Municipal Zone';
      const zoneLabel = cell.zone_name || 'Surat Region';

      rect.bindTooltip(
        `<div style="font-family: 'Inter', sans-serif; font-size: 12px; line-height: 1.45; color: #f8fafc; padding: 2px; min-width: 170px;">
          <div style="display:flex; justify-content:space-between; align-items:center; gap: 8px; margin-bottom: 2px;">
            <strong style="color:#38bdf8; font-size: 13px;">📍 ${areaLabel}</strong>
            ${cell.is_hotspot ? '<span style="background:#ef4444; color:#fff; padding:1px 5px; border-radius:3px; font-size:10px; font-weight:700;">HOTSPOT</span>' : ''}
          </div>
          <div style="font-size:10.5px; color:#94a3b8; text-transform:uppercase; margin-bottom: 4px;">
            ${zoneLabel} · Cell #${cell.cell_id}
          </div>
          <div style="font-size: 12.5px;">Observed LST: <strong style="color:#f97316;">${cell.lst_c.toFixed(1)}°C</strong></div>
          <div style="font-size: 11.5px;">Thermal Anomaly: <strong style="color:${cell.anomaly_c > 0 ? '#ef4444' : '#10b981'};">${cell.anomaly_c > 0 ? '+' : ''}${cell.anomaly_c.toFixed(1)}°C</strong></div>
          <div style="font-size:11px; color:#94a3b8; margin-top:3px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 2px;">
            Built-up: ${(cell.built_up_fraction * 100).toFixed(0)}% · NDVI: ${cell.ndvi.toFixed(2)}
          </div>
        </div>`,
        { sticky: true, opacity: 0.96 }
      );

      // Click / Touch event
      rect.on('click', () => {
        onSelectCell(cell);
      });

      gridGroup.addLayer(rect);
    });
  }, [cells, selectedCell, showHotspotsOnly, layerMode, onSelectCell]);

  // 7. Update Area Markers & Badges on Map
  useEffect(() => {
    if (!mapInstanceRef.current || !labelLayerGroupRef.current) return;
    const labelGroup = labelLayerGroupRef.current;
    labelGroup.clearLayers();

    if (!showAreaLabels) return;

    areaStats.forEach((area) => {
      const isSelected = selectedCell && Math.hypot(selectedCell.lat - area.lat, (selectedCell.lon - area.lon) * 0.93) < 0.02;

      const markerHtml = `
        <div class="area-map-badge ${area.isHotspot ? 'is-hotspot' : ''}" style="${isSelected ? 'border-color: #38bdf8 !important; transform: translate(-50%, -55%) scale(1.12);' : ''}">
          <span style="font-size: 12px;">${area.isHotspot ? '🔥' : '📍'}</span>
          <span style="font-size: 11px; font-weight: 700; color: #f8fafc;">${area.name}</span>
          <span style="font-size: 10px; font-weight: 700; color: ${area.isHotspot ? '#fca5a5' : '#38bdf8'}; background: rgba(0,0,0,0.35); padding: 1px 5px; border-radius: 4px;">
            ${area.peakLst.toFixed(1)}°C
          </span>
        </div>
      `;

      const customIcon = L.divIcon({
        className: 'custom-area-marker',
        html: markerHtml,
        iconSize: [0, 0],
        iconAnchor: [0, 0],
      });

      const marker = L.marker([area.lat, area.lon], { icon: customIcon });

      marker.on('click', () => {
        handleFlyToArea(area);
      });

      labelGroup.addLayer(marker);
    });
  }, [areaStats, showAreaLabels, selectedCell]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: '540px', display: 'flex', flexDirection: 'column' }}>
      {/* Quick-Access Top Hotspots Strip */}
      <div
        className="glass-panel"
        style={{
          position: 'absolute',
          top: '12px',
          left: '56px',
          right: '180px',
          zIndex: 999,
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '6px 12px',
          background: 'rgba(10, 15, 29, 0.90)',
          boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
          overflowX: 'auto',
          scrollbarWidth: 'none',
        }}
      >
        <span style={{ fontSize: '0.74rem', fontWeight: 700, color: '#f87171', whiteSpace: 'nowrap', display: 'flex', alignItems: 'center', gap: '4px' }}>
          🔥 KEY HOTSPOTS:
        </span>
        {topHotspots.map((area) => {
          const isActive = selectedCell && Math.hypot(selectedCell.lat - area.lat, (selectedCell.lon - area.lon) * 0.93) < 0.02;
          return (
            <button
              key={area.id}
              onClick={() => handleFlyToArea(area)}
              className={`quick-hotspot-pill ${isActive ? 'active' : ''}`}
              title={`Click to inspect ${area.name} (${area.zone})`}
            >
              <span>{area.name}</span>
              <span style={{ color: '#ef4444' }}>{area.peakLst.toFixed(1)}°C</span>
            </button>
          );
        })}
      </div>

      {/* Map Container */}
      <div ref={mapContainerRef} style={{ width: '100%', height: '100%', borderRadius: '14px' }} />

      {/* Layer Mode & Controls Toolbar (Top-Right under Hotspot Filter) */}
      <div
        className="glass-panel"
        style={{
          position: 'absolute',
          top: '56px',
          right: '14px',
          zIndex: 999,
          display: 'flex',
          flexDirection: 'column',
          gap: '6px',
          padding: '8px',
          background: 'rgba(10, 15, 29, 0.90)',
          boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
        }}
      >
        <div style={{ fontSize: '0.70rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '2px' }}>
          Thermal Render Mode
        </div>
        <div style={{ display: 'flex', gap: '4px' }}>
          <button
            onClick={() => setLayerMode('heatmap')}
            style={{
              background: layerMode === 'heatmap' ? 'rgba(239, 68, 68, 0.25)' : 'transparent',
              color: layerMode === 'heatmap' ? '#fca5a5' : 'var(--text-secondary)',
              border: layerMode === 'heatmap' ? '1px solid rgba(239, 68, 68, 0.5)' : '1px solid transparent',
              padding: '5px 8px',
              borderRadius: '6px',
              fontSize: '0.75rem',
              fontWeight: 600,
              cursor: 'pointer',
            }}
            title="Continuous smooth heat gradient (Reference style)"
          >
            🔥 Heatmap
          </button>
          <button
            onClick={() => setLayerMode('combined')}
            style={{
              background: layerMode === 'combined' ? 'rgba(56, 189, 248, 0.25)' : 'transparent',
              color: layerMode === 'combined' ? 'var(--accent-cyan)' : 'var(--text-secondary)',
              border: layerMode === 'combined' ? '1px solid rgba(56, 189, 248, 0.5)' : '1px solid transparent',
              padding: '5px 8px',
              borderRadius: '6px',
              fontSize: '0.75rem',
              fontWeight: 600,
              cursor: 'pointer',
            }}
            title="Heatmap gradient + interactive grid borders"
          >
            ✨ Combined
          </button>
          <button
            onClick={() => setLayerMode('grid')}
            style={{
              background: layerMode === 'grid' ? 'rgba(16, 185, 129, 0.25)' : 'transparent',
              color: layerMode === 'grid' ? 'var(--accent-emerald)' : 'var(--text-secondary)',
              border: layerMode === 'grid' ? '1px solid rgba(16, 185, 129, 0.5)' : '1px solid transparent',
              padding: '5px 8px',
              borderRadius: '6px',
              fontSize: '0.75rem',
              fontWeight: 600,
              cursor: 'pointer',
            }}
            title="100m analytical raster polygons"
          >
            🟩 100m Grid
          </button>
        </div>

        <div style={{ height: '1px', background: 'var(--border-subtle)', margin: '2px 0' }} />

        {/* Area Labels Toggle */}
        <button
          onClick={() => setShowAreaLabels(!showAreaLabels)}
          style={{
            background: showAreaLabels ? 'rgba(99, 102, 241, 0.2)' : 'transparent',
            color: showAreaLabels ? '#a5b4fc' : 'var(--text-muted)',
            border: showAreaLabels ? '1px solid rgba(99, 102, 241, 0.4)' : '1px solid transparent',
            padding: '5px 8px',
            borderRadius: '6px',
            fontSize: '0.75rem',
            fontWeight: 600,
            cursor: 'pointer',
            textAlign: 'center',
          }}
        >
          {showAreaLabels ? '🏷️ Area Labels: ON' : '🏷️ Area Labels: OFF'}
        </button>
      </div>

      {/* Basemap Switcher (Bottom toolbar near zoom) */}
      <div
        className="glass-panel"
        style={{
          position: 'absolute',
          bottom: '24px',
          right: '14px',
          zIndex: 999,
          display: 'flex',
          gap: '4px',
          padding: '4px',
          background: 'rgba(10, 15, 29, 0.90)',
          boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
        }}
      >
        {(Object.keys(TILE_PROVIDERS) as TileKey[]).map((key) => (
          <button
            key={key}
            onClick={() => handleTileChange(key)}
            style={{
              background: activeTile === key ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
              color: activeTile === key ? 'var(--accent-cyan)' : 'var(--text-secondary)',
              border: activeTile === key ? '1px solid rgba(56, 189, 248, 0.4)' : '1px solid transparent',
              padding: '5px 8px',
              borderRadius: '6px',
              fontSize: '0.75rem',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.2s ease',
            }}
          >
            {TILE_PROVIDERS[key].name}
          </button>
        ))}

        <div style={{ width: '1px', background: 'var(--border-subtle)', margin: '2px 2px' }} />

        <button
          onClick={handleFitSurat}
          title="Fit viewport to entire Surat municipal and metropolitan boundary"
          style={{
            background: 'rgba(99, 102, 241, 0.15)',
            color: '#a5b4fc',
            border: '1px solid rgba(99, 102, 241, 0.35)',
            padding: '5px 8px',
            borderRadius: '6px',
            fontSize: '0.75rem',
            fontWeight: 600,
            cursor: 'pointer',
          }}
        >
          🎯 Surat Extent
        </button>
      </div>

      {/* Floating Thermal Scale Legend (Bottom-Left) */}
      <div
        className="glass-panel"
        style={{
          position: 'absolute',
          bottom: '24px',
          left: '24px',
          zIndex: 999,
          padding: '12px 16px',
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
          minWidth: '220px',
          background: 'rgba(10, 15, 29, 0.90)',
        }}
      >
        <div style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
          Land Surface Temp (°C)
        </div>
        <div className="temp-gauge" style={{ width: '100%' }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          <span>&lt;30°C</span>
          <span>34°C</span>
          <span>38°C</span>
          <span>&gt;42°C</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
          <span style={{ width: '12px', height: '12px', borderRadius: '3px', background: '#ef4444', border: '1.5px solid #fff' }}></span>
          <span>Red Marker = High Vulnerability Hotspot</span>
        </div>
      </div>
    </div>
  );
};
