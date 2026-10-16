import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import { GridCell } from '../types';

interface MapViewProps {
  cells: GridCell[];
  selectedCell: GridCell | null;
  onSelectCell: (cell: GridCell) => void;
  showHotspotsOnly: boolean;
}

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

  // Initialize Leaflet Map once
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    // Center on Surat City, Gujarat
    const map = L.map(mapContainerRef.current, {
      center: [21.190, 72.825],
      zoom: 13,
      zoomControl: true,
    });

    // Dark theme CartoDB basemap
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
      subdomains: 'abcd',
      maxZoom: 19,
    }).addTo(map);

    const layerGroup = L.layerGroup().addTo(map);
    layerGroupRef.current = layerGroup;
    mapInstanceRef.current = map;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Update cell polygons when season or filter changes
  useEffect(() => {
    if (!mapInstanceRef.current || !layerGroupRef.current) return;

    layerGroupRef.current.clearLayers();

    const displayCells = showHotspotsOnly ? cells.filter((c) => c.is_hotspot) : cells;

    displayCells.forEach((cell) => {
      const color = getTemperatureColor(cell.lst_c);
      const isSelected = selectedCell?.cell_id === cell.cell_id;
      const size = 0.0045; // Approx 100m in lat/lon degrees

      // 100m grid cell polygon bounds
      const bounds: L.LatLngBoundsExpression = [
        [cell.lat - size, cell.lon - size],
        [cell.lat + size, cell.lon + size],
      ];

      const rect = L.rectangle(bounds, {
        color: isSelected ? '#38bdf8' : (cell.is_hotspot ? '#ef4444' : color),
        weight: isSelected ? 3 : (cell.is_hotspot ? 2 : 1),
        fillColor: color,
        fillOpacity: isSelected ? 0.85 : 0.65,
      });

      // Tooltip on hover
      rect.bindTooltip(
        `<div style="font-family: sans-serif; font-size: 12px; line-height: 1.4;">
          <strong>Cell #${cell.cell_id}</strong> ${cell.is_hotspot ? '<span style="color:#ef4444;">[HOTSPOT]</span>' : ''}<br/>
          Observed LST: <strong>${cell.lst_c}°C</strong><br/>
          Anomaly: <strong>${cell.anomaly_c > 0 ? '+' : ''}${cell.anomaly_c}°C</strong><br/>
          Built-up: ${(cell.built_up_fraction * 100).toFixed(0)}% | NDVI: ${cell.ndvi.toFixed(2)}
        </div>`,
        { sticky: true, opacity: 0.95 }
      );

      // Click event
      rect.on('click', () => {
        onSelectCell(cell);
      });

      layerGroupRef.current?.addLayer(rect);
    });
  }, [cells, selectedCell, showHotspotsOnly, onSelectCell]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: '520px' }}>
      <div ref={mapContainerRef} style={{ width: '100%', height: '100%', borderRadius: '14px' }} />

      {/* Floating Map Legend */}
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
          <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: '#ef4444', border: '1px solid #fff' }}></span>
          <span>Red Outline: High Anomaly Hotspot</span>
        </div>
      </div>
    </div>
  );
};
