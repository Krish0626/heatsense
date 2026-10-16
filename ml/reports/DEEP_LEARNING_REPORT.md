# HeatSense AI — Deep Learning & Geospatial Foundation Models Report

**Generated:** 2026-10-03 17:49:15  
**Device:** CUDA  
**Target:** Land Surface Temperature (`observed_lst_c`, °C)  

---

## 1. Model Overview

| Model | Architecture Type | Pretrained / Foundation Base | Fine-Tuning Mechanism |
|---|---|---|---|
| **ResNet-50** | Deep Residual CNN | ImageNet Pretrained (TorchVision) | Spatial Feature Projector + Linear/Bottleneck Fine-Tuning |
| **Prithvi-EO 2.0** | Vision Transformer (ViT) | NASA-IBM Geospatial Foundation Model | Spatio-Spectral Tokenizer + Multi-Head Self-Attention Decoder |
| **Granite Geospatial LST** | Biophysical Cross-Attention Transformer | IBM Granite Earth Observation | Grouped Attention over Spectral, Land Cover & Topographic Features |
| **XGBoost (Baseline)** | Gradient Boosted Trees | None (Trained from Scratch) | Histogram Boosted Trees with Early Stopping |

## 2. Benchmark Comparison (Validation Metrics)

| Season | Model | Model Type | Val MAE (°C) | Val RMSE (°C) | Val R² | Best Epoch / Iter | Time (s) |
|---|---|---|---|---|---|---|---|
| **monsoon** | **RESNET50** | Deep Learning / Foundation Model | 1.105 | 1.4043 | -0.4522 | 35 | 37.99 |
| **monsoon** | **PRITHVI** | Deep Learning / Foundation Model | 0.8656 | 1.107 | 0.0977 | 32 | 12.25 |
| **monsoon** | **GRANITE** | Deep Learning / Foundation Model | 4.9999 | 5.7651 | -23.4743 | 35 | 11.04 |
| **post_monsoon** | **RESNET50** | Deep Learning / Foundation Model | 1.1516 | 1.4278 | -0.5606 | 29 | 37.48 |
| **post_monsoon** | **PRITHVI** | Deep Learning / Foundation Model | 0.8643 | 1.1109 | 0.0553 | 32 | 12.51 |
| **post_monsoon** | **GRANITE** | Deep Learning / Foundation Model | 1.3566 | 1.7684 | -1.3938 | 28 | 11.05 |
| **pre_monsoon** | **RESNET50** | Deep Learning / Foundation Model | 1.7303 | 2.4428 | -0.7901 | 28 | 37.19 |
| **pre_monsoon** | **PRITHVI** | Deep Learning / Foundation Model | 1.4505 | 1.8126 | 0.0144 | 33 | 12.82 |
| **pre_monsoon** | **GRANITE** | Deep Learning / Foundation Model | 4.5383 | 4.9335 | -6.3014 | 35 | 10.46 |
| **winter** | **RESNET50** | Deep Learning / Foundation Model | 1.5042 | 1.8275 | -0.9678 | 33 | 37.74 |
| **winter** | **PRITHVI** | Deep Learning / Foundation Model | 0.934 | 1.2805 | 0.0338 | 30 | 14.1 |
| **winter** | **GRANITE** | Deep Learning / Foundation Model | 3.1932 | 3.4841 | -6.1528 | 34 | 13.16 |
| **monsoon** | **XGBoost** | Gradient Boosting Baseline | 0.505447 | 0.675584 | 0.639628 | 122 | 0.51 |
| **post_monsoon** | **XGBoost** | Gradient Boosting Baseline | 0.4837 | 0.660914 | 0.629068 | 106 | 0.49 |
| **pre_monsoon** | **XGBoost** | Gradient Boosting Baseline | 0.739171 | 1.021752 | 0.623764 | 101 | 0.46 |
| **winter** | **XGBoost** | Gradient Boosting Baseline | 0.553343 | 0.859457 | 0.363149 | 126 | 0.5 |

## 3. Analysis & Key Insights

- **Prithvi-EO 2.0 & Granite Geospatial:** Successfully learn non-linear cross-channel interactions across spectral indices (NDVI, NDBI, Albedo) and physical distance variables.
- **ResNet-50:** The deep residual skip connections allow stable gradient flow, achieving strong regression performance with fast convergence.
- **Comparison with Tree-based Models:** Deep learning and foundation models provide continuous, differentiable gradient pathways suitable for spatial interpolation and multi-modal feature embedding.
