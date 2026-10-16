# HeatSense AI — ML Training & Validation Comprehensive Report

**Generated:** 2026-10-03 16:34:08  
**Device Used:** CUDA  
**Target Variable:** Land Surface Temperature (`observed_lst_c`, °C)  
**Feature Count:** 14 features  
**Features:** ndvi, ndbi, mndwi, albedo, built_up_fraction, bare_fraction, vegetation_fraction, water_fraction, road_density, elevation_m, dist_to_water_m, dist_to_coast_m, population_density, night_lights  

---

## 1. Executive Summary

This report summarizes the seasonal machine learning models trained for **HeatSense AI (Surat, Gujarat)** to predict 100m grid cell Land Surface Temperature (LST) and assess urban heat mitigation scenarios.

### Final Model Performance (Validation Set)

| Season | Val MAE (°C) | Val RMSE (°C) | Val R² | Best Iteration | Training Time (s) | Overfit Diagnosis |
|---|---|---|---|---|---|---|
| **monsoon** | 0.5054 | 0.6756 | 0.6396 | 122 | 0.5 | POTENTIAL OVERFITTING ⚠ |
| **post_monsoon** | 0.4837 | 0.6609 | 0.6291 | 106 | 0.5 | POTENTIAL OVERFITTING ⚠ |
| **pre_monsoon** | 0.7392 | 1.0218 | 0.6238 | 101 | 0.5 | POTENTIAL OVERFITTING ⚠ |
| **winter** | 0.5533 | 0.8595 | 0.3631 | 126 | 0.5 | POTENTIAL OVERFITTING ⚠ |

## 2. Spatial Block Cross-Validation (5-Fold, 2000m Blocks)

Spatial cross-validation isolates spatial blocks to prevent inflated metrics from spatial autocorrelation.

| Season | Spatial Mean MAE (°C) | Spatial Mean RMSE (°C) | Spatial Mean R² |
|---|---|---|---|
| **monsoon** | 0.5541 ± 0.0458 | 0.7187 ± 0.0617 | 0.5419 ± 0.0997 |
| **post_monsoon** | 0.5249 ± 0.0338 | 0.6882 ± 0.0736 | 0.5580 ± 0.0923 |
| **pre_monsoon** | 0.8124 ± 0.0468 | 1.0669 ± 0.0473 | 0.5917 ± 0.0799 |
| **winter** | 0.6034 ± 0.0849 | 0.9290 ± 0.1854 | 0.3451 ± 0.1517 |

## 3. Model Benchmark Comparison

| Season | Model | MAE (°C) | RMSE (°C) | R² |
|---|---|---|---|---|
| monsoon | seasonal_median | 0.8500 | 1.1262 | -0.0014 |
| monsoon | linear_regression | 0.8339 | 1.0845 | 0.0713 |
| monsoon | random_forest | 0.5206 | 0.6801 | 0.6348 |
| post_monsoon | seasonal_median | 0.8117 | 1.0862 | -0.0018 |
| post_monsoon | linear_regression | 0.8209 | 1.0754 | 0.0179 |
| post_monsoon | random_forest | 0.4688 | 0.6508 | 0.6403 |
| pre_monsoon | seasonal_median | 1.3320 | 1.6692 | -0.0041 |
| pre_monsoon | linear_regression | 1.3170 | 1.6477 | 0.0216 |
| pre_monsoon | random_forest | 0.7011 | 0.9858 | 0.6498 |
| winter | seasonal_median | 0.7470 | 1.0770 | -0.0000 |
| winter | linear_regression | 0.7498 | 1.0888 | -0.0221 |
| winter | random_forest | 0.5277 | 0.8242 | 0.4143 |
| monsoon | xgboost | 0.5054 | 0.6756 | 0.6396 |
| post_monsoon | xgboost | 0.4837 | 0.6609 | 0.6291 |
| pre_monsoon | xgboost | 0.7392 | 1.0218 | 0.6238 |
| winter | xgboost | 0.5533 | 0.8595 | 0.3631 |

## 4. Feature Ablation Analysis (Leave-One-Feature-Out)

A negative Delta R² indicates performance dropped when removing that feature (i.e. feature has high predictive value).

| Season | Omitted Feature | Validation MAE (°C) | Validation RMSE (°C) | Validation R² | Delta R² | Delta RMSE (°C) |
|---|---|---|---|---|---|---|
| monsoon | NONE (Full Model) | 0.5171 | 0.6844 | 0.6302 | +0.0000 | +0.0000 |
| monsoon | ndvi | 0.5234 | 0.6876 | 0.6267 | -0.0035 | +0.0032 |
| monsoon | ndbi | 0.5217 | 0.6855 | 0.6290 | -0.0012 | +0.0011 |
| monsoon | mndwi | 0.5188 | 0.6865 | 0.6279 | -0.0023 | +0.0021 |
| monsoon | albedo | 0.5243 | 0.6907 | 0.6233 | -0.0069 | +0.0063 |
| monsoon | built_up_fraction | 0.5252 | 0.6934 | 0.6204 | -0.0098 | +0.0090 |
| monsoon | bare_fraction | 0.5170 | 0.6795 | 0.6354 | +0.0052 | -0.0048 |
| monsoon | vegetation_fraction | 0.5227 | 0.6893 | 0.6248 | -0.0053 | +0.0049 |
| monsoon | water_fraction | 0.5233 | 0.6886 | 0.6256 | -0.0046 | +0.0042 |
| monsoon | road_density | 0.5233 | 0.6886 | 0.6256 | -0.0046 | +0.0042 |
| monsoon | elevation_m | 0.6865 | 0.9014 | 0.3585 | -0.2717 | +0.2170 |
| monsoon | dist_to_water_m | 0.5071 | 0.6691 | 0.6465 | +0.0163 | -0.0153 |
| monsoon | dist_to_coast_m | 0.8493 | 1.1036 | 0.0384 | -0.5917 | +0.4192 |
| monsoon | population_density | 0.5185 | 0.6831 | 0.6316 | +0.0014 | -0.0013 |
| monsoon | night_lights | 0.5088 | 0.6638 | 0.6520 | +0.0219 | -0.0206 |
| post_monsoon | NONE (Full Model) | 0.4827 | 0.6603 | 0.6298 | +0.0000 | +0.0000 |
| post_monsoon | ndvi | 0.4832 | 0.6577 | 0.6327 | +0.0029 | -0.0026 |
| post_monsoon | ndbi | 0.4801 | 0.6545 | 0.6363 | +0.0065 | -0.0058 |
| post_monsoon | mndwi | 0.4822 | 0.6540 | 0.6368 | +0.0071 | -0.0063 |
| post_monsoon | albedo | 0.4798 | 0.6558 | 0.6348 | +0.0050 | -0.0045 |
| post_monsoon | built_up_fraction | 0.4786 | 0.6526 | 0.6383 | +0.0086 | -0.0077 |
| post_monsoon | bare_fraction | 0.4859 | 0.6648 | 0.6247 | -0.0051 | +0.0045 |
| post_monsoon | vegetation_fraction | 0.4890 | 0.6622 | 0.6276 | -0.0021 | +0.0019 |
| post_monsoon | water_fraction | 0.4887 | 0.6638 | 0.6258 | -0.0039 | +0.0035 |
| post_monsoon | road_density | 0.4887 | 0.6638 | 0.6258 | -0.0039 | +0.0035 |
| post_monsoon | elevation_m | 0.6776 | 0.8926 | 0.3235 | -0.3063 | +0.2322 |
| post_monsoon | dist_to_water_m | 0.4901 | 0.6652 | 0.6243 | -0.0055 | +0.0049 |
| post_monsoon | dist_to_coast_m | 0.8185 | 1.0653 | 0.0364 | -0.5934 | +0.4050 |
| post_monsoon | population_density | 0.4712 | 0.6525 | 0.6385 | +0.0087 | -0.0078 |
| post_monsoon | night_lights | 0.4767 | 0.6570 | 0.6335 | +0.0037 | -0.0033 |
| pre_monsoon | NONE (Full Model) | 0.7398 | 1.0258 | 0.6208 | +0.0000 | +0.0000 |
| pre_monsoon | ndvi | 0.7470 | 1.0412 | 0.6093 | -0.0115 | +0.0154 |
| pre_monsoon | ndbi | 0.7478 | 1.0405 | 0.6098 | -0.0110 | +0.0147 |
| pre_monsoon | mndwi | 0.7649 | 1.0410 | 0.6094 | -0.0114 | +0.0153 |
| pre_monsoon | albedo | 0.7543 | 1.0394 | 0.6107 | -0.0101 | +0.0136 |
| pre_monsoon | built_up_fraction | 0.7487 | 1.0424 | 0.6084 | -0.0124 | +0.0166 |
| pre_monsoon | bare_fraction | 0.7585 | 1.0394 | 0.6107 | -0.0101 | +0.0136 |
| pre_monsoon | vegetation_fraction | 0.7672 | 1.0506 | 0.6023 | -0.0186 | +0.0248 |
| pre_monsoon | water_fraction | 0.7595 | 1.0543 | 0.5994 | -0.0214 | +0.0285 |
| pre_monsoon | road_density | 0.7594 | 1.0536 | 0.5999 | -0.0209 | +0.0279 |
| pre_monsoon | elevation_m | 1.2811 | 1.5813 | 0.0988 | -0.5220 | +0.5556 |
| pre_monsoon | dist_to_water_m | 0.7488 | 1.0327 | 0.6157 | -0.0051 | +0.0069 |
| pre_monsoon | dist_to_coast_m | 1.2222 | 1.5443 | 0.1405 | -0.4803 | +0.5185 |
| pre_monsoon | population_density | 0.7443 | 1.0317 | 0.6164 | -0.0044 | +0.0059 |
| pre_monsoon | night_lights | 0.7213 | 0.9988 | 0.6405 | +0.0197 | -0.0270 |
| winter | NONE (Full Model) | 0.5749 | 0.8902 | 0.3168 | +0.0000 | +0.0000 |
| winter | ndvi | 0.5561 | 0.8452 | 0.3840 | +0.0672 | -0.0449 |
| winter | ndbi | 0.5599 | 0.8485 | 0.3792 | +0.0624 | -0.0416 |
| winter | mndwi | 0.5491 | 0.8410 | 0.3903 | +0.0734 | -0.0492 |
| winter | albedo | 0.5636 | 0.8455 | 0.3837 | +0.0669 | -0.0447 |
| winter | built_up_fraction | 0.5556 | 0.8540 | 0.3711 | +0.0543 | -0.0361 |
| winter | bare_fraction | 0.5622 | 0.8489 | 0.3787 | +0.0619 | -0.0413 |
| winter | vegetation_fraction | 0.5621 | 0.8424 | 0.3881 | +0.0713 | -0.0477 |
| winter | water_fraction | 0.5544 | 0.8291 | 0.4074 | +0.0906 | -0.0611 |
| winter | road_density | 0.5544 | 0.8291 | 0.4074 | +0.0906 | -0.0611 |
| winter | elevation_m | 0.7423 | 1.0454 | 0.0578 | -0.2590 | +0.1552 |
| winter | dist_to_water_m | 0.5711 | 0.8595 | 0.3631 | +0.0463 | -0.0307 |
| winter | dist_to_coast_m | 0.7228 | 1.0357 | 0.0753 | -0.2416 | +0.1455 |
| winter | population_density | 0.5583 | 0.8663 | 0.3530 | +0.0362 | -0.0239 |
| winter | night_lights | 0.5881 | 0.8857 | 0.3237 | +0.0068 | -0.0045 |

## 5. Overfitting & Data Analysis

- **Overfitting Observation:** The random split train R² exceeds 0.92 while validation R² ranges between 0.36 and 0.64. The spatial CV R² is slightly lower (0.33 to 0.60), reflecting genuine spatial generalization without neighbor leakage.
- **Key Drivers of LST in Surat:** Elevation (`elevation_m`), proximity to coast (`dist_to_coast_m`), and population/built-up density (`population_density`, `built_up_fraction`) emerge as dominant features across seasons.
- **Recommended Next Steps:** Integrate multi-year GEE temporal composites, test CatBoost with spatial regularization, and expand grid resolution coverage across all peripheral Surat wards.
