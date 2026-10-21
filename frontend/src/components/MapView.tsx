import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import { GridCell } from '../types';

interface MapViewProps {
  cells: GridCell[];
  selectedCell: GridCell | null;
  onSelectCell: (cell: GridCell) => void;
  showHotspotsOnly: boolean;
}

// 100% Free Tile Providers — No API Key Required!
const TILE_PROVIDERS = {
  osm: {
    name: '🗺️ Street Map (OSM)',
    url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19,
    className: '',
  },
  satellite: {
    name: '🛰️ Satellite (Esri)',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attribution: 'Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics',
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

// Function to calculate color from temperature in °C
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
  const layerGroupRef = useRef<L.LayerGroup | null>(null);
  const tileLayerRef = useRef<L.TileLayer | null>(null);
  const [activeTile, setActiveTile] = useState<TileKey>('satellite');

  // Initialize Leaflet Map once
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    // Center on Surat City, Gujarat (Tapi River, Ring Road, Athwa, Varachha)
    const map = L.map(mapContainerRef.current, {
      center: [21.192, 72.830],
      zoom: 13,
      zoomControl: true,
    });

    // Add initial base tile layer (Satellite by default for stunning remote sensing context)
    const provider = TILE_PROVIDERS['satellite'];
    const tileLayer = L.tileLayer(provider.url, {
      attribution: provider.attribution,
      maxZoom: provider.maxZoom,
      className: provider.className,
    }).addTo(map);

    tileLayerRef.current = tileLayer;

    const layerGroup = L.layerGroup().addTo(map);
    layerGroupRef.current = layerGroup;
    mapInstanceRef.current = map;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Handle Tile Provider Switching
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

    // Ensure cell polygons stay on top of basemap
    if (layerGroupRef.current) {
      layerGroupRef.current.eachLayer((layer: any) => {
        if (typeof layer.bringToFront === 'function') {
          layer.bringToFront();
        }
      });
    }

    tileLayerRef.current = newTileLayer;
  };

  // Update cell polygons when season, cells, or filters change
  useEffect(() => {
    if (!mapInstanceRef.current || !layerGroupRef.current) return;

    layerGroupRef.current.clearLayers();

    const displayCells = showHotspotsOnly ? cells.filter((c) => c.is_hotspot) : cells;

    displayCells.forEach((cell) => {
      const color = getTemperatureColor(cell.lst_c);
      const isSelected = selectedCell?.cell_id === cell.cell_id;
      const size = 0.0055; // 100m-scale grid cell representation

      // 100m grid cell polygon bounds
      const bounds: L.LatLngBoundsExpression = [
        [cell.lat - size, cell.lon - size],
        [cell.lat + size, cell.lon + size],
      ];

      const rect = L.rectangle(bounds, {
        color: isSelected ? '#38bdf8' : (cell.is_hotspot ? '#ef4444' : color),
        weight: isSelected ? 3.5 : (cell.is_hotspot ? 2.5 : 1.2),
        fillColor: color,
        fillOpacity: isSelected ? 0.88 : (cell.is_hotspot ? 0.78 : 0.65),
      });

      // Tooltip on hover
      rect.bindTooltip(
        `<div style="font-family: 'Inter', sans-serif; font-size: 12px; line-height: 1.45; color: #f8fafc; padding: 2px;">
          <div style="display:flex; justify-content:space-between; align-items:center; gap: 8px; margin-bottom: 3px;">
            <strong style="color:#38bdf8;">Cell #${cell.cell_id}</strong>
            ${cell.is_hotspot ? '<span style="background:#ef4444; color:#fff; padding:1px 5px; border-radius:3px; font-size:10px; font-weight:700;">HOTSPOT</span>' : ''}
          </div>
          <div>Observed LST: <strong style="color:#f97316;">${cell.lst_c.toFixed(1)}°C</strong></div>
          <div>Thermal Anomaly: <strong style="color:${cell.anomaly_c > 0 ? '#ef4444' : '#10b981'};">${cell.anomaly_c > 0 ? '+' : ''}${cell.anomaly_c.toFixed(1)}°C</strong></div>
          <div style="font-size:11px; color:#94a3b8; margin-top:2px;">Built-up: ${(cell.built_up_fraction * 100).toFixed(0)}% · NDVI: ${cell.ndvi.toFixed(2)}</div>
        </div>`,
        { sticky: true, opacity: 0.96 }
      );

      // Click event to select cell
      rect.on('click', () => {
        onSelectCell(cell);
      });

      layerGroupRef.current?.addLayer(rect);
    });
  }, [cells, selectedCell, showHotspotsOnly, onSelectCell]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: '520px' }}>
      <div ref={mapContainerRef} style={{ width: '100%', height: '100%', borderRadius: '14px' }} />

      {/* Basemap Switcher (Top-Left under zoom buttons) */}
      <div
        className="glass-panel"
        style={{
          position: 'absolute',
          top: '14px',
          left: '56px',
          zIndex: 999,
          display: 'flex',
          gap: '4px',
          padding: '4px',
          background: 'rgba(10, 15, 29, 0.88)',
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
              padding: '6px 10px',
              borderRadius: '6px',
              fontSize: '0.78rem',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.2s ease',
            }}
          >
            {TILE_PROVIDERS[key].name}
          </button>
        ))}
      </div>

      {/* Floating Map Legend (Bottom-Left) */}
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
          background: 'rgba(10, 15, 29, 0.88)',
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
          <span>Red Border = High Anomaly Hotspot</span>
        </div>
      </div>
    </div>
  );
};
