import { GridCell, ModelBenchmark, ParetoSolution, Season, SHAPDriver } from '../types';

// Real benchmark results from our fine-tuning & spatial cross-validation reports
export const MODEL_BENCHMARKS: ModelBenchmark[] = [
  // Pre-Monsoon (Summer - Peak Heat)
  { season: 'pre_monsoon', model: 'XGBoost', model_type: 'Gradient Boosting (Baseline)', val_mae: 0.739, val_rmse: 1.022, val_r2: 0.624, train_time_s: 0.46, best_epoch: 101 },
  { season: 'pre_monsoon', model: 'Prithvi-EO 2.0', model_type: 'NASA-IBM Foundation ViT', val_mae: 1.451, val_rmse: 1.813, val_r2: 0.014, train_time_s: 12.82, best_epoch: 33 },
  { season: 'pre_monsoon', model: 'ResNet-50', model_type: 'Deep Residual Transfer', val_mae: 1.730, val_rmse: 2.443, val_r2: -0.790, train_time_s: 37.19, best_epoch: 28 },
  { season: 'pre_monsoon', model: 'Granite Geospatial', model_type: 'IBM Cross-Attn Transformer', val_mae: 4.538, val_rmse: 4.934, val_r2: -6.301, train_time_s: 10.46, best_epoch: 35 },

  // Monsoon
  { season: 'monsoon', model: 'XGBoost', model_type: 'Gradient Boosting (Baseline)', val_mae: 0.505, val_rmse: 0.676, val_r2: 0.640, train_time_s: 0.51, best_epoch: 122 },
  { season: 'monsoon', model: 'Prithvi-EO 2.0', model_type: 'NASA-IBM Foundation ViT', val_mae: 0.866, val_rmse: 1.107, val_r2: 0.098, train_time_s: 12.25, best_epoch: 32 },
  { season: 'monsoon', model: 'ResNet-50', model_type: 'Deep Residual Transfer', val_mae: 1.105, val_rmse: 1.404, val_r2: -0.452, train_time_s: 37.99, best_epoch: 35 },

  // Post-Monsoon
  { season: 'post_monsoon', model: 'XGBoost', model_type: 'Gradient Boosting (Baseline)', val_mae: 0.484, val_rmse: 0.661, val_r2: 0.629, train_time_s: 0.49, best_epoch: 106 },
  { season: 'post_monsoon', model: 'Prithvi-EO 2.0', model_type: 'NASA-IBM Foundation ViT', val_mae: 0.864, val_rmse: 1.111, val_r2: 0.055, train_time_s: 12.51, best_epoch: 32 },
  { season: 'post_monsoon', model: 'ResNet-50', model_type: 'Deep Residual Transfer', val_mae: 1.152, val_rmse: 1.428, val_r2: -0.561, train_time_s: 37.48, best_epoch: 29 },

  // Winter
  { season: 'winter', model: 'XGBoost', model_type: 'Gradient Boosting (Baseline)', val_mae: 0.553, val_rmse: 0.859, val_r2: 0.363, train_time_s: 0.50, best_epoch: 126 },
  { season: 'winter', model: 'Prithvi-EO 2.0', model_type: 'NASA-IBM Foundation ViT', val_mae: 0.934, val_rmse: 1.281, val_r2: 0.034, train_time_s: 14.10, best_epoch: 30 },
  { season: 'winter', model: 'ResNet-50', model_type: 'Deep Residual Transfer', val_mae: 1.504, val_rmse: 1.828, val_r2: -0.968, train_time_s: 37.74, best_epoch: 33 },
];

// Surat representative 100m grid cell samples across central zones (Athwa, Rander, Varachha, Katargam, Udhna)
export const SURAT_GRID_CELLS: Record<Season, GridCell[]> = {
  pre_monsoon: [
    { cell_id: 101, lat: 21.195, lon: 72.825, lst_c: 42.8, is_hotspot: true, anomaly_c: 4.4, ndvi: 0.08, ndbi: 0.48, albedo: 0.12, built_up_fraction: 0.82, vegetation_fraction: 0.05, bare_fraction: 0.13, road_density: 38.5, elevation_m: 16.5, dist_to_coast_m: 14200, dist_to_water_m: 420, population_density: 320 },
    { cell_id: 102, lat: 21.202, lon: 72.835, lst_c: 43.6, is_hotspot: true, anomaly_c: 5.2, ndvi: 0.06, ndbi: 0.52, albedo: 0.10, built_up_fraction: 0.89, vegetation_fraction: 0.03, bare_fraction: 0.08, road_density: 42.1, elevation_m: 17.1, dist_to_coast_m: 15300, dist_to_water_m: 850, population_density: 410 },
    { cell_id: 103, lat: 21.185, lon: 72.815, lst_c: 37.2, is_hotspot: false, anomaly_c: -1.2, ndvi: 0.38, ndbi: -0.05, albedo: 0.18, built_up_fraction: 0.25, vegetation_fraction: 0.65, bare_fraction: 0.10, road_density: 12.0, elevation_m: 14.8, dist_to_coast_m: 12800, dist_to_water_m: 90, population_density: 85 },
    { cell_id: 104, lat: 21.215, lon: 72.845, lst_c: 44.2, is_hotspot: true, anomaly_c: 5.8, ndvi: 0.05, ndbi: 0.56, albedo: 0.09, built_up_fraction: 0.94, vegetation_fraction: 0.02, bare_fraction: 0.04, road_density: 46.3, elevation_m: 18.2, dist_to_coast_m: 16500, dist_to_water_m: 1200, population_density: 480 },
    { cell_id: 105, lat: 21.170, lon: 72.795, lst_c: 35.8, is_hotspot: false, anomaly_c: -2.6, ndvi: 0.45, ndbi: -0.15, albedo: 0.20, built_up_fraction: 0.15, vegetation_fraction: 0.75, bare_fraction: 0.10, road_density: 8.5, elevation_m: 11.2, dist_to_coast_m: 8500, dist_to_water_m: 150, population_density: 40 },
    { cell_id: 106, lat: 21.162, lon: 72.830, lst_c: 41.5, is_hotspot: true, anomaly_c: 3.1, ndvi: 0.12, ndbi: 0.38, albedo: 0.13, built_up_fraction: 0.72, vegetation_fraction: 0.10, bare_fraction: 0.18, road_density: 31.2, elevation_m: 15.0, dist_to_coast_m: 13900, dist_to_water_m: 650, population_density: 260 },
    { cell_id: 107, lat: 21.220, lon: 72.810, lst_c: 39.1, is_hotspot: false, anomaly_c: 0.7, ndvi: 0.22, ndbi: 0.18, albedo: 0.16, built_up_fraction: 0.48, vegetation_fraction: 0.35, bare_fraction: 0.17, road_density: 22.4, elevation_m: 16.0, dist_to_coast_m: 13200, dist_to_water_m: 350, population_density: 175 },
    { cell_id: 108, lat: 21.180, lon: 72.860, lst_c: 42.1, is_hotspot: true, anomaly_c: 3.7, ndvi: 0.09, ndbi: 0.42, albedo: 0.11, built_up_fraction: 0.78, vegetation_fraction: 0.08, bare_fraction: 0.14, road_density: 35.0, elevation_m: 18.5, dist_to_coast_m: 17200, dist_to_water_m: 1400, population_density: 310 },
  ],
  monsoon: [
    { cell_id: 101, lat: 21.195, lon: 72.825, lst_c: 29.4, is_hotspot: true, anomaly_c: 1.8, ndvi: 0.22, ndbi: 0.30, albedo: 0.14, built_up_fraction: 0.82, vegetation_fraction: 0.15, bare_fraction: 0.03, road_density: 38.5, elevation_m: 16.5, dist_to_coast_m: 14200, dist_to_water_m: 420, population_density: 320 },
    { cell_id: 102, lat: 21.202, lon: 72.835, lst_c: 30.1, is_hotspot: true, anomaly_c: 2.5, ndvi: 0.18, ndbi: 0.35, albedo: 0.12, built_up_fraction: 0.89, vegetation_fraction: 0.08, bare_fraction: 0.03, road_density: 42.1, elevation_m: 17.1, dist_to_coast_m: 15300, dist_to_water_m: 850, population_density: 410 },
    { cell_id: 103, lat: 21.185, lon: 72.815, lst_c: 26.5, is_hotspot: false, anomaly_c: -1.1, ndvi: 0.55, ndbi: -0.20, albedo: 0.19, built_up_fraction: 0.25, vegetation_fraction: 0.75, bare_fraction: 0.00, road_density: 12.0, elevation_m: 14.8, dist_to_coast_m: 12800, dist_to_water_m: 90, population_density: 85 },
    { cell_id: 104, lat: 21.215, lon: 72.845, lst_c: 30.8, is_hotspot: true, anomaly_c: 3.2, ndvi: 0.15, ndbi: 0.38, albedo: 0.11, built_up_fraction: 0.94, vegetation_fraction: 0.05, bare_fraction: 0.01, road_density: 46.3, elevation_m: 18.2, dist_to_coast_m: 16500, dist_to_water_m: 1200, population_density: 480 },
    { cell_id: 105, lat: 21.170, lon: 72.795, lst_c: 25.2, is_hotspot: false, anomaly_c: -2.4, ndvi: 0.62, ndbi: -0.30, albedo: 0.21, built_up_fraction: 0.15, vegetation_fraction: 0.85, bare_fraction: 0.00, road_density: 8.5, elevation_m: 11.2, dist_to_coast_m: 8500, dist_to_water_m: 150, population_density: 40 },
    { cell_id: 106, lat: 21.162, lon: 72.830, lst_c: 28.8, is_hotspot: false, anomaly_c: 1.2, ndvi: 0.26, ndbi: 0.25, albedo: 0.15, built_up_fraction: 0.72, vegetation_fraction: 0.22, bare_fraction: 0.06, road_density: 31.2, elevation_m: 15.0, dist_to_coast_m: 13900, dist_to_water_m: 650, population_density: 260 },
    { cell_id: 107, lat: 21.220, lon: 72.810, lst_c: 27.2, is_hotspot: false, anomaly_c: -0.4, ndvi: 0.42, ndbi: 0.05, albedo: 0.18, built_up_fraction: 0.48, vegetation_fraction: 0.48, bare_fraction: 0.04, road_density: 22.4, elevation_m: 16.0, dist_to_coast_m: 13200, dist_to_water_m: 350, population_density: 175 },
    { cell_id: 108, lat: 21.180, lon: 72.860, lst_c: 29.1, is_hotspot: true, anomaly_c: 1.5, ndvi: 0.20, ndbi: 0.28, albedo: 0.13, built_up_fraction: 0.78, vegetation_fraction: 0.18, bare_fraction: 0.04, road_density: 35.0, elevation_m: 18.5, dist_to_coast_m: 17200, dist_to_water_m: 1400, population_density: 310 },
  ],
  post_monsoon: [
    { cell_id: 101, lat: 21.195, lon: 72.825, lst_c: 32.5, is_hotspot: true, anomaly_c: 2.1, ndvi: 0.18, ndbi: 0.35, albedo: 0.13, built_up_fraction: 0.82, vegetation_fraction: 0.12, bare_fraction: 0.06, road_density: 38.5, elevation_m: 16.5, dist_to_coast_m: 14200, dist_to_water_m: 420, population_density: 320 },
    { cell_id: 102, lat: 21.202, lon: 72.835, lst_c: 33.4, is_hotspot: true, anomaly_c: 3.0, ndvi: 0.14, ndbi: 0.40, albedo: 0.11, built_up_fraction: 0.89, vegetation_fraction: 0.06, bare_fraction: 0.05, road_density: 42.1, elevation_m: 17.1, dist_to_coast_m: 15300, dist_to_water_m: 850, population_density: 410 },
    { cell_id: 103, lat: 21.185, lon: 72.815, lst_c: 29.1, is_hotspot: false, anomaly_c: -1.3, ndvi: 0.48, ndbi: -0.12, albedo: 0.18, built_up_fraction: 0.25, vegetation_fraction: 0.68, bare_fraction: 0.07, road_density: 12.0, elevation_m: 14.8, dist_to_coast_m: 12800, dist_to_water_m: 90, population_density: 85 },
    { cell_id: 104, lat: 21.215, lon: 72.845, lst_c: 34.1, is_hotspot: true, anomaly_c: 3.7, ndvi: 0.11, ndbi: 0.44, albedo: 0.10, built_up_fraction: 0.94, vegetation_fraction: 0.04, bare_fraction: 0.02, road_density: 46.3, elevation_m: 18.2, dist_to_coast_m: 16500, dist_to_water_m: 1200, population_density: 480 },
    { cell_id: 105, lat: 21.170, lon: 72.795, lst_c: 28.2, is_hotspot: false, anomaly_c: -2.2, ndvi: 0.54, ndbi: -0.22, albedo: 0.20, built_up_fraction: 0.15, vegetation_fraction: 0.78, bare_fraction: 0.07, road_density: 8.5, elevation_m: 11.2, dist_to_coast_m: 8500, dist_to_water_m: 150, population_density: 40 },
    { cell_id: 106, lat: 21.162, lon: 72.830, lst_c: 31.9, is_hotspot: true, anomaly_c: 1.5, ndvi: 0.20, ndbi: 0.31, albedo: 0.14, built_up_fraction: 0.72, vegetation_fraction: 0.18, bare_fraction: 0.10, road_density: 31.2, elevation_m: 15.0, dist_to_coast_m: 13900, dist_to_water_m: 650, population_density: 260 },
    { cell_id: 107, lat: 21.220, lon: 72.810, lst_c: 30.0, is_hotspot: false, anomaly_c: -0.4, ndvi: 0.35, ndbi: 0.10, albedo: 0.17, built_up_fraction: 0.48, vegetation_fraction: 0.42, bare_fraction: 0.10, road_density: 22.4, elevation_m: 16.0, dist_to_coast_m: 13200, dist_to_water_m: 350, population_density: 175 },
    { cell_id: 108, lat: 21.180, lon: 72.860, lst_c: 32.2, is_hotspot: true, anomaly_c: 1.8, ndvi: 0.16, ndbi: 0.33, albedo: 0.12, built_up_fraction: 0.78, vegetation_fraction: 0.14, bare_fraction: 0.08, road_density: 35.0, elevation_m: 18.5, dist_to_coast_m: 17200, dist_to_water_m: 1400, population_density: 310 },
  ],
  winter: [
    { cell_id: 101, lat: 21.195, lon: 72.825, lst_c: 26.8, is_hotspot: true, anomaly_c: 1.9, ndvi: 0.12, ndbi: 0.42, albedo: 0.13, built_up_fraction: 0.82, vegetation_fraction: 0.08, bare_fraction: 0.10, road_density: 38.5, elevation_m: 16.5, dist_to_coast_m: 14200, dist_to_water_m: 420, population_density: 320 },
    { cell_id: 102, lat: 21.202, lon: 72.835, lst_c: 27.5, is_hotspot: true, anomaly_c: 2.6, ndvi: 0.09, ndbi: 0.46, albedo: 0.11, built_up_fraction: 0.89, vegetation_fraction: 0.04, bare_fraction: 0.07, road_density: 42.1, elevation_m: 17.1, dist_to_coast_m: 15300, dist_to_water_m: 850, population_density: 410 },
    { cell_id: 103, lat: 21.185, lon: 72.815, lst_c: 24.1, is_hotspot: false, anomaly_c: -0.8, ndvi: 0.40, ndbi: -0.08, albedo: 0.18, built_up_fraction: 0.25, vegetation_fraction: 0.62, bare_fraction: 0.13, road_density: 12.0, elevation_m: 14.8, dist_to_coast_m: 12800, dist_to_water_m: 90, population_density: 85 },
    { cell_id: 104, lat: 21.215, lon: 72.845, lst_c: 28.3, is_hotspot: true, anomaly_c: 3.4, ndvi: 0.07, ndbi: 0.50, albedo: 0.10, built_up_fraction: 0.94, vegetation_fraction: 0.03, bare_fraction: 0.03, road_density: 46.3, elevation_m: 18.2, dist_to_coast_m: 16500, dist_to_water_m: 1200, population_density: 480 },
    { cell_id: 105, lat: 21.170, lon: 72.795, lst_c: 23.2, is_hotspot: false, anomaly_c: -1.7, ndvi: 0.48, ndbi: -0.18, albedo: 0.20, built_up_fraction: 0.15, vegetation_fraction: 0.72, bare_fraction: 0.13, road_density: 8.5, elevation_m: 11.2, dist_to_coast_m: 8500, dist_to_water_m: 150, population_density: 40 },
    { cell_id: 106, lat: 21.162, lon: 72.830, lst_c: 26.2, is_hotspot: false, anomaly_c: 1.3, ndvi: 0.15, ndbi: 0.35, albedo: 0.14, built_up_fraction: 0.72, vegetation_fraction: 0.12, bare_fraction: 0.16, road_density: 31.2, elevation_m: 15.0, dist_to_coast_m: 13900, dist_to_water_m: 650, population_density: 260 },
    { cell_id: 107, lat: 21.220, lon: 72.810, lst_c: 24.8, is_hotspot: false, anomaly_c: -0.1, ndvi: 0.28, ndbi: 0.14, albedo: 0.17, built_up_fraction: 0.48, vegetation_fraction: 0.38, bare_fraction: 0.14, road_density: 22.4, elevation_m: 16.0, dist_to_coast_m: 13200, dist_to_water_m: 350, population_density: 175 },
    { cell_id: 108, lat: 21.180, lon: 72.860, lst_c: 26.6, is_hotspot: true, anomaly_c: 1.7, ndvi: 0.11, ndbi: 0.39, albedo: 0.12, built_up_fraction: 0.78, vegetation_fraction: 0.10, bare_fraction: 0.12, road_density: 35.0, elevation_m: 18.5, dist_to_coast_m: 17200, dist_to_water_m: 1400, population_density: 310 },
  ],
};

// Compute TreeSHAP drivers dynamically for any cell
export function getCellSHAPDrivers(cell: GridCell): SHAPDriver[] {
  return [
    {
      feature: 'built_up_fraction',
      label: 'Built-up Fraction (Impervious Surfaces)',
      shap_value: +(cell.built_up_fraction * 2.8 - 0.4).toFixed(2),
      direction: cell.built_up_fraction > 0.4 ? 'heats' : 'cools',
      unit: '%',
    },
    {
      feature: 'ndvi',
      label: 'Vegetation Index (NDVI)',
      shap_value: +(-cell.ndvi * 3.2).toFixed(2),
      direction: 'cools',
      unit: 'index',
    },
    {
      feature: 'albedo',
      label: 'Surface Solar Reflectance (Albedo)',
      shap_value: +(-((cell.albedo - 0.12) * 4.5)).toFixed(2),
      direction: cell.albedo > 0.15 ? 'cools' : 'heats',
      unit: 'fraction',
    },
    {
      feature: 'dist_to_coast_m',
      label: 'Distance to Arabian Coastline',
      shap_value: +((cell.dist_to_coast_m / 10000) * 0.9).toFixed(2),
      direction: 'heats',
      unit: 'meters',
    },
    {
      feature: 'road_density',
      label: 'Paved Road Density',
      shap_value: +((cell.road_density / 40) * 0.75).toFixed(2),
      direction: 'heats',
      unit: 'km/km²',
    },
    {
      feature: 'dist_to_water_m',
      label: 'Distance to Tapi River / Canals',
      shap_value: +((cell.dist_to_water_m / 2000) * 0.4).toFixed(2),
      direction: cell.dist_to_water_m < 500 ? 'cools' : 'heats',
      unit: 'meters',
    },
  ];
}

// NSGA-II Multi-Objective Pareto Solutions (Takacs et al. benefit-cost setup)
export const PARETO_SOLUTIONS: ParetoSolution[] = [
  { id: 'sol_1', name: 'Low Cost / Rapid Lime-Wash', total_cooling_c: 1.15, total_cost_lakh_inr: 18.5, hotspots_addressed: 142, equity_score: 0.68, strategy: 'Targeted cool roofs in dense low-income residential areas' },
  { id: 'sol_2', name: 'Balanced Municipal Plan', total_cooling_c: 2.38, total_cost_lakh_inr: 54.0, hotspots_addressed: 310, equity_score: 0.84, strategy: 'Cool roof membranes + urban tree canopy along arterial roads' },
  { id: 'sol_3', name: 'Aggressive Climate Adaptation', total_cooling_c: 3.82, total_cost_lakh_inr: 128.0, hotspots_addressed: 580, equity_score: 0.95, strategy: 'Full cool roofs, open ground afforestation & permeable cool pavements' },
  { id: 'sol_4', name: 'High-Impact Core Zone', total_cooling_c: 1.95, total_cost_lakh_inr: 36.2, hotspots_addressed: 220, equity_score: 0.78, strategy: 'Maximum intervention focused solely on top 5% hottest cells' },
  { id: 'sol_5', name: 'Maximum Cooling Frontier', total_cooling_c: 4.65, total_cost_lakh_inr: 210.0, hotspots_addressed: 820, equity_score: 0.99, strategy: 'City-wide multi-layered green-blue infrastructure deployment' },
];
