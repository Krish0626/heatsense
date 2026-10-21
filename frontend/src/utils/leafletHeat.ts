import L from 'leaflet';

/**
 * High-performance, self-contained Leaflet Heatmap Layer
 * Inspired by Vladimir Agafonkin's Leaflet.heat and simpleheat.
 * Runs natively in Vite without CommonJS hoisting or window.L issues.
 */

interface HeatmapOptions extends L.LayerOptions {
  minOpacity?: number;
  maxZoom?: number;
  max?: number;
  radius?: number;
  blur?: number;
  gradient?: Record<number, string>;
}

class SimpleHeatCanvas {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private width: number;
  private height: number;
  private maxVal: number = 1.0;
  private data: Array<[number, number, number]> = [];
  private circleCanvas: HTMLCanvasElement | null = null;
  private circleRadius: number = 25;
  private circleBlur: number = 15;
  private gradData: Uint8ClampedArray | null = null;

  defaultRadius = 25;
  defaultGradient: Record<number, string> = {
    0.2: '#3b82f6',
    0.4: '#06b6d4',
    0.6: '#10b981',
    0.75: '#facc15',
    0.88: '#f97316',
    1.0: '#ef4444',
  };

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;
    this.width = canvas.width;
    this.height = canvas.height;
  }

  setData(data: Array<[number, number, number]>) {
    this.data = data;
    return this;
  }

  setMax(max: number) {
    this.maxVal = max;
    return this;
  }

  setDimensions(width: number, height: number) {
    this.width = this.canvas.width = width;
    this.height = this.canvas.height = height;
    return this;
  }

  setRadius(r: number, blur?: number) {
    const targetBlur = blur !== undefined ? blur : 15;
    if (this.circleRadius === r && this.circleBlur === targetBlur && this.circleCanvas) {
      return this;
    }
    this.circleRadius = r;
    this.circleBlur = targetBlur;

    const circle = (this.circleCanvas = document.createElement('canvas'));
    const ctx = circle.getContext('2d')!;
    const totalR = r + this.circleBlur;

    circle.width = circle.height = 2 * totalR;

    ctx.shadowOffsetX = ctx.shadowOffsetY = 200;
    ctx.shadowBlur = this.circleBlur;
    ctx.shadowColor = 'black';

    ctx.beginPath();
    ctx.arc(totalR - 200, totalR - 200, r, 0, 2 * Math.PI, true);
    ctx.closePath();
    ctx.fill();

    return this;
  }

  setGradient(grad: Record<number, string>) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    const gradient = ctx.createLinearGradient(0, 0, 0, 256);

    canvas.width = 1;
    canvas.height = 256;

    for (const stop in grad) {
      gradient.addColorStop(+stop, grad[stop]);
    }

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 1, 256);

    this.gradData = ctx.getImageData(0, 0, 1, 256).data;
    return this;
  }

  draw(_minOpacity: number = 0.05) {
    if (!this.circleCanvas) {
      this.setRadius(this.defaultRadius, 15);
    }
    if (!this.gradData) {
      this.setGradient(this.defaultGradient);
    }

    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    const totalR = this.circleRadius + this.circleBlur;

    for (const point of this.data) {
      const [x, y, intensity] = point;
      if (intensity < 0.02) continue;
      // Weight each point gently so only dense clusters of hot cells accumulate to deep red
      ctx.globalAlpha = Math.min(0.42, Math.pow(intensity / this.maxVal, 1.4) * 0.42);
      ctx.drawImage(this.circleCanvas!, x - totalR, y - totalR);
    }

    const image = ctx.getImageData(0, 0, this.width, this.height);
    const pixels = image.data;
    const grad = this.gradData!;

    for (let i = 0, len = pixels.length; i < len; i += 4) {
      const alpha = pixels[i + 3];
      if (alpha > 0) {
        const offset = Math.min(alpha, 255) * 4;
        pixels[i] = grad[offset] || 0;
        pixels[i + 1] = grad[offset + 1] || 0;
        pixels[i + 2] = grad[offset + 2] || 0;
        // Soft edge blending matching reference Image 2
        pixels[i + 3] = Math.round(Math.min(alpha * 0.88, 220));
      }
    }

    ctx.putImageData(image, 0, 0);
    return this;
  }
}

export class ThermalHeatLayer extends L.Layer {
  private latlngs: Array<[number, number, number]>;
  public override options: HeatmapOptions;
  private canvas: HTMLCanvasElement | null = null;
  private heat: SimpleHeatCanvas | null = null;
  private animFrame: number | null = null;

  constructor(latlngs: Array<[number, number, number]>, options: HeatmapOptions = {}) {
    super();
    this.latlngs = latlngs;
    this.options = {
      minOpacity: 0.25,
      maxZoom: 18,
      max: 1.0,
      radius: 32,
      blur: 24,
      gradient: {
        0.15: '#3b82f6', // Cool Blue
        0.35: '#06b6d4', // Cyan
        0.55: '#10b981', // Emerald Green
        0.70: '#facc15', // Amber / Yellow
        0.85: '#f97316', // Deep Orange
        1.00: '#ef4444', // Burning Red Hotspot
      },
      ...options,
    };
  }

  setLatLngs(latlngs: Array<[number, number, number]>) {
    this.latlngs = latlngs;
    this.redraw();
    return this;
  }

  onAdd(map: L.Map): this {
    const canvas = (this.canvas = L.DomUtil.create('canvas', 'leaflet-heatmap-layer') as HTMLCanvasElement);
    const size = map.getSize();
    canvas.width = size.x;
    canvas.height = size.y;

    canvas.style.position = 'absolute';
    canvas.style.left = '0';
    canvas.style.top = '0';
    canvas.style.pointerEvents = 'none';

    map.getPanes().overlayPane.appendChild(canvas);

    this.heat = new SimpleHeatCanvas(canvas);
    this.heat.setRadius(this.options.radius || 32, this.options.blur || 24);
    if (this.options.gradient) {
      this.heat.setGradient(this.options.gradient);
    }
    if (this.options.max) {
      this.heat.setMax(this.options.max);
    }

    map.on('moveend', this.reset, this);
    map.on('zoomend', this.reset, this);

    this.reset();
    return this;
  }

  onRemove(map: L.Map): this {
    if (this.canvas && this.canvas.parentNode) {
      this.canvas.parentNode.removeChild(this.canvas);
    }

    map.off('moveend', this.reset, this);
    map.off('zoomend', this.reset, this);

    if (this.animFrame !== null) {
      cancelAnimationFrame(this.animFrame);
      this.animFrame = null;
    }
    return this;
  }

  redraw() {
    if (!this.heat || this.animFrame !== null) return;
    this.animFrame = requestAnimationFrame(() => {
      this.animFrame = null;
      this.draw();
    });
  }

  private reset = () => {
    if (!this.canvas || !(this as any)._map) return;
    const map = (this as any)._map as L.Map;
    const topLeft = map.containerPointToLayerPoint([0, 0]);
    L.DomUtil.setPosition(this.canvas, topLeft);

    const size = map.getSize();
    if (this.canvas.width !== size.x || this.canvas.height !== size.y) {
      this.heat?.setDimensions(size.x, size.y);
    }
    this.draw();
  };

  private draw() {
    if (!this.heat || !(this as any)._map) return;
    const map = (this as any)._map as L.Map;
    const size = map.getSize();

    // Dynamically calculate cell spacing in screen pixels to scale radius & blur
    const pCenter = map.latLngToContainerPoint([21.175, 72.835]);
    const pNeighbor = map.latLngToContainerPoint([21.175 + 0.0089, 72.835 + 0.0090]);
    const stepDist = Math.hypot(pNeighbor.x - pCenter.x, pNeighbor.y - pCenter.y) || 45;

    // Radius tuned so points in a hotspot form a smooth organic plume without flooding the entire city
    const dynamicRadius = Math.max(26, Math.round(stepDist * 1.15));
    const dynamicBlur = Math.max(22, Math.round(stepDist * 0.95));
    this.heat.setRadius(dynamicRadius, dynamicBlur);

    const totalRadius = dynamicRadius + dynamicBlur;
    const bounds = new L.Bounds(L.point([-totalRadius, -totalRadius]), size.add([totalRadius, totalRadius]));

    const points: Array<[number, number, number]> = [];

    for (const [lat, lon, val] of this.latlngs) {
      const p = map.latLngToContainerPoint([lat, lon]);
      if (bounds.contains(p)) {
        points.push([Math.round(p.x), Math.round(p.y), val]);
      }
    }

    this.heat.setData(points);
    this.heat.draw(this.options.minOpacity ?? 0.0);
  }
}
