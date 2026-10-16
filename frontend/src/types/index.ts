export type Season = 'pre_monsoon' | 'monsoon' | 'post_monsoon' | 'winter';

export interface GridCell {
  cell_id: number;
  lat: number;
  lon: number;
  lst_c: number;
  is_hotspot: boolean;
  anomaly_c: number;
  ndvi: number;
  ndbi: number;
  albedo: number;
  built_up_fraction: number;
  vegetation_fraction: number;
  bare_fraction: number;
  road_density: number;
  elevation_m: number;
  dist_to_coast_m: number;
  dist_to_water_m: number;
  population_density: number;
}

export interface SHAPDriver {
  feature: string;
  label: string;
  shap_value: number;
  direction: 'heats' | 'cools';
  unit: string;
}

export interface InterventionCap {
  name: string;
  type: 'cool_roof' | 'tree_canopy' | 'cool_pavement';
  max_sqm: number;
  current_sqm: number;
  unit_cost_inr: number; // cost per m² over 10 years
  cooling_per_100sqm: number; // °C cooling per 100 m²
}

export interface ParetoSolution {
  id: string;
  name: string;
  total_cooling_c: number;
  total_cost_lakh_inr: number;
  hotspots_addressed: number;
  equity_score: number;
  strategy: string;
}

export interface ModelBenchmark {
  season: Season;
  model: string;
  model_type: string;
  val_mae: number;
  val_rmse: number;
  val_r2: number;
  train_time_s: number;
  best_epoch: number | string;
}
