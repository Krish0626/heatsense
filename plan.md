# HeatSense AI: Working Plan & Implementation Status

**Project:** Urban Heat Mitigation & Cooling Decision Support System (Project-I, BTCO16706)  
**Target City:** Surat, Gujarat (SMC Boundaries, 100m Analytical Grid Cells)  
**Updated:** October 2026 (Post Fine-Tuning of Deep Foundation Models & Spatial Validation)  

---

## 🚀 Executive Implementation Dashboard

| Phase / Component | Status | Completed Highlights | Remaining Work |
|---|---|---|---|
| **Phase 0: Architecture & Setup** | ✅ **DONE** | Python 3.11 venv, CUDA 12.1 PyTorch, dependencies, project tree. | Maintain env and CI. |
| **Phase 1: Geospatial Data Pipeline** | ✅ **DONE** | 100m grid (`grid_cells.geojson`), 14 feature extraction (`cell_features.parquet`), 4 seasons. | Optional high-res night thermal proxy. |
| **Phase 2: ML Baseline & Spatial CV** | ✅ **DONE** | XGBoost Regressors (4 seasons), 5-Fold 2000m Spatial Block Cross-Validation. | Benchmark tuning with CatBoost / LightGBM. |
| **Phase 2b: Foundation Models & DL** | ✅ **DONE** | Fine-tuned **Prithvi-EO 2.0**, **ResNet-50**, and **Granite Geospatial** on GPU across all 4 seasons with checkpoints & diagnostic curves. | Optional patch-based 2D multi-spectral spatial transformer extension. |
| **Phase 3: Hotspot Detection & SHAP** | ✅ **DONE** | Anomaly & percentile hotspots, TreeSHAP global beeswarm, feature importances, dependence plots. | Interactive local waterfall per cell in UI. |
| **Phase 4 & 5: Feasibility & What-If** | 🔄 **IN PROGRESS** | Intervention caps logic (roofs, roads, open land), cost coefficients, counterfactual LST engine. | Connect scenario API to dynamic frontend sliders. |
| **Phase 6: Multi-Objective Optimization** | 🔄 **IN PROGRESS** | `pymoo` NSGA-II solver framework, surrogate cooling formulation, budget constraints. | End-to-end Pareto front generation API and map overlay. |
| **Phase 7: Backend & Frontend App** | 🔄 **IN PROGRESS** | FastAPI backend skeleton, Leaflet frontend setup, JWT auth, GeoJSON endpoints. | Complete interactive dashboard views, scenario simulation UI, Pareto explorer. |
| **Phase 8: Validation & Final Report** | 📋 **TODO** | Comprehensive ML reports (`ML_TRAINING_REPORT.md`, `DEEP_LEARNING_REPORT.md`) generated. | Final deck preparation, mid-sem evidence screenshots, thesis write-up. |

---

## 1. What we are building

We take one city, Surat, cut it into 100 m x 100 m cells, and work out for every cell how hot it gets in each of the four Indian seasons. Cells that run unusually hot are flagged as hotspots. For each hotspot we explain which surface properties are driving the heat (vegetation, albedo, built-up fraction, roads, land use). Then we test which cooling measures could physically fit in that cell (a rooftop, a road strip, a patch of open ground), estimate how much cooling each would buy and what it would cost, and hand the planner a ranked, budget-aware set of options.

The flow in one line: grid, features, seasonal LST model, hotspot detection, SHAP explanation, feasibility caps, what-if simulation, cost, Pareto trade-off, report.

The deck says "SMC boundaries", so Surat Municipal Corporation limits are already the assumed study area. India-wide scaling is a later, optional step. Everything below is scoped to Surat first.

---

## 2. Review of the mid-sem deck

### 2.1 What is already decided

| Area | Choice in the deck |
|---|---|
| Grid | 100 m analytical cells, hierarchy City > Zones > Planning Areas > Grid Cells |
| Data | Landsat 8/9 thermal, Sentinel-2 optical, SMC boundaries |
| Model | XGBoost regressor, seasonal LST in °C, MSE loss |
| Features | NDVI, albedo, LULC class, built-up fraction, road density |
| Explainability | TreeSHAP per cell |
| Interventions | Rooftop, road, open ground, limited by physical land caps |
| Optimization | NSGA-II, marked partial/planned |
| Backend | FastAPI, Pydantic v2, JWT, PostgreSQL + PostGIS, GeoAlchemy2 |
| Frontend | React 19, TypeScript, Vite, Leaflet |

### 2.2 Things worth fixing or deciding early

1. Agenda item 08 is "Implementation Status & Evidence", but the exported PDF has no slide for it (page numbers skip 2, 11 and 14). Check that screenshots of the running app are in the final deck.
2. In the database slide, `is_hotspot` and `building_rooftop_sqm` sit on `planning_areas`. Hotspots are defined per cell and per season, and rooftop area is a per-cell quantity. Move both to `grid_cells` (rooftop area) and `heat_predictions` (hotspot flag per season).
3. `feasibility_caps / recommendations` is one box. Caps are inputs and recommendations are outputs, so they should be separate tables. Scenario runs and Pareto solutions also need their own tables (see section 9).
4. "Predict seasonal LST" is ambiguous. If the target is the observed seasonal LST of a cell and the inputs are surface descriptors from the same period, the model is an explanatory model, not a forecaster. That is fine for SHAP, but say so in the report. A forecast of future LST is a different project.
5. SHAP explains the model, not the physics. A cell's high SHAP value for built-up fraction means the model links it to heat, not that removing buildings would cool it by that amount. The cooling estimate in the what-if step needs its own logic (section 7.4) and should be labelled as an estimate.
6. Random train/test splits on grid cells will look far better than reality because neighbouring cells are near copies of each other. Use spatial block cross-validation (section 6.4).
7. Daytime satellite LST is a surface temperature at a late-morning overpass. The health problem in Surat is mostly at night (section 4.3). We should say this openly and, if time allows, add a night-time proxy.
8. Monsoon (June to September) is cloudy. Expect gaps in optical data for that season and plan compositing across several years.

---

## 3. Base papers and what each gives us

The three references in the deck are the method papers. The rest of this section is the literature that sits closest to our exact pipeline, found during this sweep.

### 3.1 Base papers from the deck

| Paper | Used for |
|---|---|
| Voogt & Oke (2003), Thermal remote sensing of urban climates, Remote Sensing of Environment 86(3) | Definitions of surface UHI, what satellite thermal data can and cannot say |
| Lundberg & Lee (2017), A unified approach to interpreting model predictions (SHAP), NeurIPS. arXiv 1705.07874 | Shapley value framework |
| Chen & Guestrin (2016), XGBoost, ACM SIGKDD. arXiv 1603.02754 | The regressor |

Add two more to the reference list because we use them directly: Lundberg et al. (2018), Consistent individualized feature attribution for tree ensembles (the TreeSHAP algorithm, arXiv 1802.03888), and Deb et al. (2002), A fast and elitist multiobjective genetic algorithm: NSGA-II, IEEE Transactions on Evolutionary Computation.

### 3.2 Closest work to our pipeline (XGBoost + SHAP for urban heat)

- Hotspot mapping with XGBoost and SHAP during heat waves (2025). XGBoost reached about 0.925 accuracy, and the top drivers were NDBI, built-up density, green space density, elevation and distance to water. Good template for our hotspot classifier and feature list. https://pubmed.ncbi.nlm.nih.gov/40810789/
- Urban form and surface temperature with XGBoost-SHAP (2025). NDBI, night lights, NDVI and elevation together explained about 61% of the contribution, and effects were nonlinear: NDVI cooling gets stronger once it passes roughly 0.5. These threshold effects are exactly what SHAP dependence plots should show for Surat. https://pubmed.ncbi.nlm.nih.gov/40380825/
- Nonlinear drivers of urban LST in Wuhan (Frontiers in Environmental Science, 2026). Compares six regressors (MLR, SVR, AdaBoost, XGBoost, CatBoost, LightGBM) with cross-validation and then applies SHAP. A ready-made model comparison protocol. https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2026.1801455/full
- XAI across eight US cities (Sustainability, 2026). Models both LST and SUHII (relative urban-minus-rural intensity) and uses city-held-out cross-validation. XGBoost was best for SUHII (R² 0.879). Useful for deciding what our target should be. https://doi.org/10.3390/su18083694
- XGBoost-SHAP across local climate zones, Shinjuku, Tokyo (2025). R² 0.921 against Landsat-referenced LST, with partial dependence plots. https://www.sciencedirect.com/science/article/abs/pii/S2210670725009485
- LightGBM for urban LST with interpretability (PMC, 2025). Reports R² of 0.85 to 0.92 and quotes 0.83 to 0.87 for XGBoost and deep nets in other work. So a realistic R² target for us is 0.85 to 0.92. Anything above 0.97 should make us suspect leakage. https://pmc.ncbi.nlm.nih.gov/articles/PMC11859931/

### 3.3 Closest work to the optimization module (ML + NSGA-II for cooling)

- Beijing vacant-land re-greening (2026). CatBoost with explainable AI finds LST drivers, then NSGA-II trades off overall cooling, cooling in the hottest areas, and ecological connectivity. Strategic allocation beat random allocation by 12.9% on overall cooling and 64.8% on targeted cooling. This is the closest match to our plan, including the idea of adding an equity objective. https://www.sciencedirect.com/science/article/abs/pii/S1618866726000944
- Machine-learning-based greening optimization for tropical cities (2026). Notes that pure NSGA-II often ignores land availability, budgets and phasing, which is exactly why our feasibility caps matter. https://www.sciencedirect.com/science/article/abs/pii/S036013232600274X
- Adaptive cooling networks in Shanghai (2025). Couples an adaptive NSGA-III with interpretable XGBoost. https://www.sciencedirect.com/science/article/abs/pii/S0360132325009977
- Blue-green space optimization with NSGA-II on grid-based decision units (Scientific Reports, 2026). Shows the grid-cell-as-decision-unit setup working. https://www.nature.com/articles/s41598-026-59100-x

### 3.4 Surat and India specific

- IRADe, Climate Adaptive and Gender Integrated Heat Wave Action Plan of Surat City (APN-GCR). Built on Surat hotspot maps, with slums and low-income settlements found to sit in warmer-than-average areas. This is our local benchmark and our motivation for a vulnerability layer. https://www.apn-gcr.org/wp-content/uploads/2023/06/Climate-Adaptive-and-Gender-Integrated-Heat-Wave-Action-Plan-of-Surat-City.pdf
- Land Surface Temperature Maps, Surat and Rajshahi (APN-GCR). Landsat 8 and 9 LST of Surat computed from emissivity and NDVI, with hotspots picked by the percentile method. We can compare our hotspot map against theirs. https://www.apn-gcr.org/wp-content/uploads/2023/06/Urban-Heat-Island-Maps-of-Surat-and-Rajshahi.pdf
- Takacs et al. (2025), Prioritizing heat adaptation measures across Indian cities: a benefit-cost analysis, npj Urban Sustainability. Covers Surat, Chennai and Lucknow at 100 m resolution and gives costs, cooling coefficients and benefit-cost ratios we can reuse (section 7). Code and data are on Zenodo. https://www.nature.com/articles/s42949-025-00271-3 and https://zenodo.org/records/14041656
- Ahmedabad SUHI study with MODIS LST, NDVI and albedo plus a 158-point field campaign, useful for a semi-arid Gujarat reference. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6749296/
- Assessment of urbanization impact on UHI and rainfall for Surat, Acta Geophysica (2022). https://link.springer.com/article/10.1007/s11600-021-00715-1
- Sabarmati basin LST and hydrometeorological variables, Gujarat. Notes that Landsat 8 thermal bands are natively 100 m and are resampled to 30 m in products, which fits our 100 m grid neatly. https://www.sciencedirect.com/science/article/pii/S2211714825000330

### 3.5 Intervention effectiveness (feeds the what-if step)

- Tree cooling meta-analysis, 182 studies, 110 cities (Communications Earth & Environment, 2024). Trees can lower pedestrian-level temperature by up to 12 °C; tree-on-roof setups give 1.8 to 4 °C at the roof. https://www.nature.com/articles/s43247-024-01908-4
- Cool and green roofs modelling (Environmental Research Letters, 2014). At full coverage, green roofs cut daytime surface UHI by roughly 4 °C but nighttime by only about 1 °C. https://iopscience.iop.org/article/10.1088/1748-9326/9/5/055002
- Cool pavements, systematic review and meta-analysis (2023). Raising pavement albedo from 0.1 to 0.3 can lower warm-season pavement surface temperature by up to 12 °C in some studies. https://www.sciencedirect.com/science/article/pii/S2212095523000640
- Review of roof mitigation (Buildings, 2025). Cool roofs come out as the most effective rooftop measure, then green roofs, then PV roofs. https://www.mdpi.com/2075-5309/15/21/3835
- Review of LST retrieval algorithms for Landsat 8 (Shanghai). Explains split-window versus single-channel and why we should just use the USGS Level-2 surface temperature product. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8198215/

---

## 4. Surat context that shapes the design

### 4.1 Climate and scale

Surat is coastal and hot semi-arid, on the Tapi river near the Gulf of Khambhat. Takacs et al. modelled it at 100 m and found night-time temperatures average 3 to 4 °C above nearby rural areas, and heat-wave days rising from about 30 a year now to nearly 50 under a high-emission scenario by 2050.

### 4.2 Population of concern

The IRADe action plan ties Surat's hotspots to slums and low-income settlements. Cool roofs (lime wash) have already been piloted in Surat, so a rooftop intervention is not a hypothetical in this city.

### 4.3 Day versus night

Takacs et al. report that the daytime UHI in Surat is minimal because of sea-breeze cooling, while the night-time UHI is the strong one. Landsat sees Surat in the late morning. So our hotspots are daytime surface hotspots. That is still valuable (surface heat drives roof and road heating, and correlates with built-up form), but the report should not claim we are mapping night-time heat stress. Options to reduce this gap are listed in section 5.4.

### 4.4 Grid size estimate

If the SMC boundary covers roughly 450 km², a 100 m grid has about 45,000 cells. VERIFY against the actual boundary polygon. This is small enough to keep the full grid in memory and to run XGBoost prediction over every cell in well under a second.

---

## 5. Data plan

### 5.1 Kaggle datasets

I could not open the Kaggle pages from my side (they returned a browser check), so the descriptions below come from search result snippets and titles. Open each one and check the file list and licence before you commit to it. None of them is a Surat 100 m dataset, so treat them as supporting material, not the training set.

| # | Dataset | Link | What it is | Fit for us |
|---|---|---|---|---|
| 1 | Land Surface Temperature (LST) over India | https://www.kaggle.com/datasets/animeshchoudhury/land-surface-temperature-lst-over-india | MODIS MOD11A1 v6, daily daytime LST at 1 km, high-quality pixels, converted to °C. Monthly medians plus seasonal medians for DJF, MAM, JJAS and ON. Listed in early 2024. | Best of the lot. The season split matches our four seasons. Use it as a coarse cross-check of our Surat seasonal means and for season definitions. Too coarse (1 km) to train the 100 m model. |
| 2 | Urban Heat Island (UHI) Monitoring Dataset | https://www.kaggle.com/datasets/atharvasoundankar/urban-heat-island-uhi-monitoring-dataset | Described as a dataset on urbanization's impact on local climate and health. Listed April 2025. Newest of the UHI sets I found. | Unknown structure. Could be tabular or synthetic. Good for prototyping the API and dashboard before real data is ready. |
| 3 | Urban Heat Island (UHI) Effect Prediction | https://www.kaggle.com/datasets/taruneshburman/urban-heat-island-uhi-effect-prediction | UHI prediction data, listed Dec 2024. | Contents not verified. Check columns and geography before use. |
| 4 | Global Urban Heat Island (UHI) Data Set | https://www.kaggle.com/datasets/riccardomogavero/global-urban-heat-island-uhi-data-set | Global UHI data, first listed 2020. | Older, likely city-level. Useful only for context. |
| 5 | Extreme-Weather Temperature Data set | https://www.kaggle.com/datasets/akheelmd4/extreme-weather-temperature-data-set | Air temperature data for heat wave and cold spell forecasting. | Low fit. Air temperature, not surface temperature. |

Outside Kaggle but worth knowing: the Sen-2 LULC dataset (213,761 Sentinel-2 RGB chips of central India with masks for water, dense forest, sparse forest, barren land, built-up, agriculture and fallow land) is on Mendeley Data at https://data.mendeley.com/datasets/f4ky6ks248/3. It is only needed if we decide to train our own land cover classifier, which I would avoid (see 5.2).

### 5.2 The datasets we will actually build from

Most of the training data has to come from Google Earth Engine (GEE), because no ready-made Kaggle file has Surat features at 100 m. Exact collection IDs below are from memory of the GEE catalog, so VERIFY each one in the catalog before writing code.

| Need | Source | Notes |
|---|---|---|
| Target: LST | Landsat 8 and 9 Collection 2 Level 2, surface temperature band (ST_B10) | USGS says there are no restrictions on Landsat product use. Native thermal pixels are 100 m, resampled to 30 m in the product. Mask clouds with the QA band. |
| NDVI, NDBI, MNDWI, albedo | Sentinel-2 surface reflectance (10 to 20 m) and Landsat optical bands | Compute per season as median composites. Landsat-derived shortwave albedo via the Liang formula, or MODIS albedo as a coarse check (the Ahmedabad study used MODIS albedo). |
| Land cover | Dynamic World (Sentinel-2, 10 m, near real time) and ESA WorldCover (10 m) | Use these rather than training a classifier. Take class fractions per cell. Dynamic World was used alongside Landsat in recent LST work. |
| Roof area | Google Open Buildings footprints | Takacs et al. used Open Buildings to size roof area for cool roof costing. Cross-check with OSM buildings. |
| Roads | OpenStreetMap (osmnx or a Geofabrik extract) | Road length density per cell and road area from width buffers. |
| Elevation | SRTM 30 m DEM | Elevation had strong influence in the hotspot studies above. |
| Distance to water and coast | Derived from land cover and OSM | Tapi river, canals, Gulf of Khambhat. |
| Weather context | ERA5-Land hourly (about 9 km) | For overpass-day air temperature and humidity as context, not as a per-cell feature. |
| Population | GHS-POP (Global Human Settlement Layer) | Used by Takacs et al. Needed for the vulnerability and equity layer. |
| Night lights | VIIRS nighttime lights (optional) | Showed up as a top driver in one XGBoost-SHAP study. |
| Boundaries | Surat Municipal Corporation ward and zone boundaries | VERIFY where SMC publishes shapefiles. If not public, trace from official maps or use OSM admin boundaries. |
| Benchmark | IRADe/APN Surat LST maps and heat action plan | Compare our hotspots to theirs. |
| Optional | ECOSTRESS thermal (about 70 m, varying overpass times including night) | Possible night-time proxy. Check availability over Surat. |
| Optional | Global UHI heat-wave exposure dataset, 1 km, 2003 to 2020 (Scientific Data, 2026) https://www.nature.com/articles/s41597-026-06877-1 | Context layer only. |

### 5.3 Study period and season definitions

Use the Indian seasons, which also match the Kaggle India LST dataset: winter (Dec to Feb), pre-monsoon (Mar to May), monsoon (Jun to Sep), post-monsoon (Oct to Nov). Pull 2019 to 2025 so every season has several usable scenes even after cloud masking. The monsoon composite will be the weakest, so report how many valid observations each cell has and keep that count as a data quality column.

### 5.4 Ways to handle the day/night gap (pick one, decide by week 6)

1. State the limitation in the report and stop. Cheapest.
2. Add a coarse night-time check from MODIS night LST (1 km) or ECOSTRESS night passes, and show whether daytime hotspots overlap night-time warm areas.
3. Add a vulnerability layer (population density, slum footprints) so that ranking blends heat exposure with who lives there. This lines up with the IRADe plan and does not require night data.

Option 3 plus a written limitation is the realistic choice for a semester project.

---

## 6. Modelling plan & Current Status

### 6.1 Grid & Data Format [✅ DONE]
- 100m analytical cells generated (`data/processed/grid_cells.geojson` & `data/processed/cell_features.parquet`).
- Stable `cell_id` with centroid coordinates and 14 standardized remote sensing features.

### 6.2 Target Variable [✅ DONE]
- Seasonal Land Surface Temperature (`observed_lst_c`, °C) computed across 4 seasons (`pre_monsoon`, `monsoon`, `post_monsoon`, `winter`).

### 6.3 Features Extracted (14 Features) [✅ DONE]
- Spectral: `ndvi`, `ndbi`, `mndwi`, `albedo`
- Land Cover: `built_up_fraction`, `bare_fraction`, `vegetation_fraction`, `water_fraction`, `road_density`
- Biophysical / Geographic: `elevation_m`, `dist_to_water_m`, `dist_to_coast_m`, `population_density`, `night_lights`

### 6.4 Baseline Machine Learning & Spatial CV [✅ DONE]
- **Models Implemented**: Seasonal Medians, Linear Regression, Random Forest, and **XGBoost Regressors** per season.
- **Validation**: 5-Fold 2000m Spatial Block Cross-Validation preventing spatial autocorrelation data leakage.
- **Metrics**: 
  - Monsoon XGBoost: Val MAE `0.505 °C`, Val RMSE `0.676 °C`, Val R² `0.640`
  - Post-Monsoon XGBoost: Val MAE `0.484 °C`, Val RMSE `0.661 °C`, Val R² `0.629`
  - Pre-Monsoon (Summer) XGBoost: Val MAE `0.739 °C`, Val RMSE `1.022 °C`, Val R² `0.624`
  - Winter XGBoost: Val MAE `0.553 °C`, Val RMSE `0.859 °C`, Val R² `0.363`

### 6.5 Deep Learning & Geospatial Foundation Models [✅ DONE]
- **Models Implemented & Fine-Tuned on GPU (CUDA)**:
  1. **ResNet-50**: Pretrained CNN with spatial feature projection + residual regression head.
  2. **Prithvi-EO 2.0**: NASA-IBM Earth Observation Vision Transformer (ViT) with spatio-spectral tokenization.
  3. **Granite Geospatial LST**: IBM Granite cross-attention transformer over biophysical feature groups.
- **Validation Results**:
  - **Prithvi-EO 2.0**: Achieved highest deep learning performance across all seasons (Monsoon Val MAE `0.866 °C`, Winter Val MAE `0.934 °C`, Summer Val MAE `1.451 °C`).
  - Saved 12 production checkpoints in [`ml/models_store/deep_learning/`](file:///c:/Users/krish/OneDrive/Desktop/heatsense/ml/models_store/deep_learning).
  - Diagnostic learning curves and actual vs predicted plots stored in [`ml/reports/plots/`](file:///c:/Users/krish/OneDrive/Desktop/heatsense/ml/reports/plots).

### 6.6 Hotspot Detection [✅ DONE]
- Percentile and anomaly-based hotspot classification per season.
- Spatial aggregation across 100m grid cells.

### 6.7 Explainability (TreeSHAP) [✅ DONE]
- Global SHAP feature importances and beeswarm plots generated for all seasons.
- Dependence plots for critical thermal drivers (`elevation_m`, `dist_to_coast_m`, `population_density`, `dist_to_water_m`).
- Feature ablation analysis confirming elevation and distance to coast as major geographic controls in Surat.

---

## 7. Feasibility, what-if and cost engine

### 7.1 Interventions to support

Start with three that map to the deck's caps and to how Indian cities are already acting, then add two if time allows.

Core: cool roof (lime wash or reflective membrane) on rooftops, tree planting and canopy on open ground, cool pavement or shading on road strips.
Stretch: green roof, small water body or pocket park.

### 7.2 Feasibility caps per cell

Cap for each intervention is a physical area limit, computed from the same layers as the features.
Rooftop cap = footprint area in the cell times a usable fraction (exclude roofs that already have vegetation or solar, and roofs too small to treat; start with a flat usable fraction like 0.6 and flag it as an assumption).
Road cap = road area from OSM centerlines and width classes, minus roads where paving is not the city's to change.
Open ground cap = bare and grass land from land cover minus land that is protected or private. This one is the hardest to get right, so start with "bare plus grass in Dynamic World" and note the limitation.

Store caps in a `feasibility_caps` table by cell and intervention, in square metres.

### 7.3 Unit costs and lifetimes (starting values)

These come from Takacs et al. (2025), who took them from an NRDC 2018 India study and standardized them to 2020 US dollars. Convert to INR with a stated rate, and replace with SMC or CPWD schedule-of-rates numbers if we can get them.

| Item | Value | Note |
|---|---|---|
| Lime wash cool roof | 0.75 USD per m² | Assumed effective for 2 years, so reapply about 5 times in a 10-year horizon |
| HDPE cool roof membrane | 2.10 USD per m² | Assumed effective for 5 years |
| Tree planting | 2.50 USD per tree, plus 5 USD per tree per year maintenance | Two trees assumed per 100 m² |
| Cool pavement | not found | TODO, get from a schedule of rates or a paper with Indian unit costs |
| Green roof | not found | TODO |

Compute cost over a fixed horizon (10 years) so short-lived options such as lime wash are compared fairly with longer-lived ones.

### 7.4 How to estimate cooling (decide this early, it is the weakest link)

There are two honest ways, and a good report shows both.

Model counterfactual. Change the cell's features to reflect the intervention (albedo up for a cool roof on a share of its area, NDVI up for planted area, built-up fraction down where paved area is replaced) and re-predict LST with the trained XGBoost. The difference is the model's estimated cooling. This is quick and spatially specific, but the model only learned associations from existing Surat cells, so it should not be trusted for feature values outside the range it has seen. Clip changes to the training range.

Literature bounds. Compare the counterfactual number with published ranges and flag any result outside them. Useful anchors: Takacs et al. found each extra 10% of tree canopy lowers average air temperature by about 0.43 °C and wet bulb globe temperature by about 0.2 °C in Indian cities, and Indian cool-roof evaluations show 2 to 5 °C lower indoor temperature. Surface temperature effects are usually larger than air temperature effects, so do not mix the two in one number. Say which one each figure refers to.

One more caution from Takacs et al.: cool roofs covering under about 20% of the urban surface have a negligible effect on city-wide air temperature, and lime wash fades. So cool roofs help the people under them much more than the neighbourhood around them. The what-if output should reflect that by reporting roof-level and cell-level effects separately if we can.

### 7.5 What the what-if endpoint returns

For a chosen cell or group of cells, season and intervention mix: area used per intervention (must respect caps), estimated LST change with an uncertainty range, total and annualized cost, cost per degree of cooling, and the constraints that were binding.

---

## 8. Optimization (NSGA-II) plan

### 8.1 Problem statement

Decision: how much area of each intervention to place in each candidate cell or planning area.
Objectives to minimize: total cost, and negative total cooling (that is, maximize cooling). Third objective to consider: cooling weighted by population or vulnerability, as an equity term, following the Beijing study.
Constraints: area in each cell and intervention at or below its cap, optional total budget.

### 8.2 Keeping it tractable

45,000 cells times three interventions gives 135,000 variables, far too many for NSGA-II to search well. Two workable reductions:
1. Only optimize over the top N hotspot cells (say 300 to 1,000).
2. Or optimize at planning-area level and allocate within each area afterwards.

Re-running XGBoost inside every fitness evaluation is wasteful. Precompute a per-cell cooling coefficient for each intervention (cooling per square metre, from the counterfactual in 7.4) and use it as a fast surrogate inside NSGA-II. Then verify the final Pareto solutions by running the full model once each.

### 8.3 Tools

`pymoo` has a clean NSGA-II implementation with constraint handling. For a sanity check, run a budget sweep with a simple greedy or linear-programming allocation (best cooling per rupee first) and confirm the NSGA-II front is at least as good. If NSGA-II cannot beat greedy on a two-objective problem, that is worth saying honestly in the report.

### 8.4 Output

A Pareto front of cost versus cooling, a few named solutions on it (cheapest useful, best value, maximum cooling), and a per-solution allocation map. The deck already shows this as an optional branch, so the API endpoint and the sequence diagram do not need to change.

---

## 9. System design updates

### 9.1 Tables to add or change

Keep the deck's `cities`, `zones`, `planning_areas`, `grid_cells`. Then:

| Table | Purpose |
|---|---|
| `data_versions` | Which imagery window, model version and feature set produced each prediction |
| `cell_features` | Seasonal feature values per cell (or JSON on `grid_cells`), including valid-observation count |
| `heat_predictions` | cell, season, observed LST, predicted LST, anomaly, is_hotspot, shap_values |
| `interventions` | Type, unit cost, lifetime, effect assumption, source |
| `feasibility_caps` | cell, intervention, cap in m², assumptions used |
| `scenarios` and `scenario_results` | Inputs and outputs of each what-if run |
| `pareto_runs` and `pareto_solutions` | Optimization runs and the non-dominated solutions |
| `users` | JWT roles, if we keep authentication |

Move `is_hotspot` and `building_rooftop_sqm` off `planning_areas` (see 2.2). Add GiST indexes on all geometry columns and B-tree indexes on `(cell_id, season)`.

### 9.2 API (matches the sequence diagram)

- GET `/api/v1/grid/cells?bbox=...&season=...` for viewport loading (FR-02)
- GET `/api/v1/grid/cells/{id}/prediction` returns LST and SHAP values
- GET `/api/v1/hotspots?season=...` returns the ranked list (FR-04)
- POST `/api/v1/scenarios/simulate` runs a what-if
- POST `/api/v1/optimization/pareto` runs NSGA-II
- GET `/api/v1/reports/{id}.csv` for export (FR-08)

Keep the model out of the request path where possible: precompute predictions and SHAP for all cells and seasons offline, and load the model in the API only for scenario simulation.

### 9.3 Frontend

Leaflet layers: LST choropleth by season, hotspot overlay, per-cell side panel with the SHAP waterfall, scenario sliders (area per intervention capped by the feasibility limit), Pareto scatter plot with a click-to-view allocation map. With about 45,000 polygons, use vector tiles or server-side viewport queries instead of loading the full grid into the browser.

---

## 10. Roadmap & Detailed Task Status

| Phase | Weeks | Work Description | Status | What is Done | What is Left |
|---|---|---|---|---|---|
| **0. Decisions & Env** | 1 | Setup project architecture, virtual env, Surat boundary & season definitions | ✅ **DONE** | Python 3.11 `.venv`, PyTorch + CUDA, directory structure, season definitions. | Automated CI/CD test triggers. |
| **1. Data Pipeline** | 1 to 3 | GEE features, remote sensing rasters, 100m grid build, cell feature extraction | ✅ **DONE** | 14 biophysical & spectral features extracted across 4 seasons in `cell_features.parquet` & `grid_cells.geojson`. | Optional real-time GEE live fetcher. |
| **2. Baseline & Spatial CV** | 3 to 4 | Baseline regressors, XGBoost seasonal training, spatial block cross-validation | ✅ **DONE** | 4 seasonal XGBoost models trained with 5-fold 2000m spatial block CV, spatial metrics logged. | Hyperparameter sweeps with CatBoost & LightGBM. |
| **2b. Deep Foundation Models** | 4 to 5 | NASA-IBM Prithvi-EO 2.0, ResNet-50 transfer learning, IBM Granite cross-attention | ✅ **DONE** | Fine-tuned across all 4 seasons on GPU with 12 `.pt` checkpoints, loss curves, and benchmark report. | Integrate PyTorch inference endpoint in FastAPI backend. |
| **3. Hotspots & SHAP** | 4 to 5 | Hotspot classification, TreeSHAP global & local attributions, feature ablation | ✅ **DONE** | Anomaly thresholding, global beeswarm, dependence plots, and ablation metrics in reports. | Fast cell-level local waterfall JSON generation on click. |
| **4. Feasibility & Costs** | 5 to 6 | Physical land caps per cell (roofs, roads, open space), unit cost database | 🔄 **IN PROGRESS** | Cap formulas defined, 10-year annualized cost model per intervention implemented. | Verify SMC local schedule of rates (SoR). |
| **5. What-If Simulation** | 6 to 8 | Counterfactual feature modification, bounded cooling estimates, uncertainty ranges | 🔄 **IN PROGRESS** | Counterfactual delta math, clipping guards, scenario evaluation function. | Connect POST `/api/v1/scenarios/simulate` endpoint to frontend. |
| **6. Multi-Objective Opt** | 8 to 10 | `pymoo` NSGA-II solver, surrogate cooling model, budget-cooling Pareto trade-offs | 🔄 **IN PROGRESS** | NSGA-II formulation on top hotspot cells, constraint handling. | Full Pareto API endpoint + frontend interactive scatter plot. |
| **7. App Integration** | 6 to 11 | FastAPI REST backend, Leaflet map dashboard, JWT auth, scenario sliders | 🔄 **IN PROGRESS** | Backend structure, DB models, GeoJSON serving, frontend Vite skeleton. | Finalize Leaflet choropleth layer switching & scenario simulator UI. |
| **8. Validation & Reports** | 11 to 14 | Model benchmark report, final documentation, project presentation deck | 📋 **TODO** | `ML_TRAINING_REPORT.md` and `DEEP_LEARNING_REPORT.md` generated. | Final mid-sem/end-sem presentation slides and project report PDF. |

A workable split for five people: one on data and GEE, one on modelling and SHAP, one on feasibility, cost and optimization, one on backend and database, one on frontend. Someone should own the report from week 1, not week 12.

---

## 11. Evaluation and what to report

Model quality: spatial-CV R², RMSE and MAE per season against held-out blocks, and against a baseline. Hotspot quality: overlap with the IRADe Surat hotspot map, and consistency across years. Cross-check of seasonal means against the Kaggle India MODIS LST dataset aggregated to 1 km. Explanation quality: do the top SHAP drivers agree with the literature (NDBI, built-up density and low vegetation as the main heating drivers)? Intervention estimates: are the model counterfactual numbers inside the published ranges? Optimization: does NSGA-II match or beat a greedy allocation, and how does the front change with different caps or costs? System: API latency for viewport queries and scenario runs, test coverage on core services.

An ablation table (drop each feature family and record the change in error) makes the SHAP story much stronger and is cheap to produce.

---

## 12. Risks

| Risk | Effect | Response |
|---|---|---|
| Monsoon cloud gaps | Weak or missing monsoon LST | Multi-year median, report valid-observation counts, allow "low confidence" flag |
| Spatial autocorrelation | Inflated accuracy | Spatial block CV only |
| Correlated features | Confusing SHAP | Group features into driver families, drop redundant ones |
| Cooling estimates outside the training range | Wrong recommendations | Clip counterfactuals, compare to literature ranges, show uncertainty |
| Unit costs are 2020 USD and from a review, not from SMC | Cost figures off | Label as indicative, swap in schedule-of-rates numbers when found |
| Day-only view of heat | Mismatch with night-time health risk | State it, add vulnerability layer |
| NSGA-II search too large | Poor fronts | Restrict to top hotspots, use surrogate, compare with greedy |
| Data licences and access | Blocked steps | Landsat is unrestricted; check licences on Open Buildings, OSM (ODbL) and any Kaggle file |
| Scope creep to multi-city | Nothing finishes | Surat only until phase 8 |

---

## 13. Open questions for the team

1. Is the SMC boundary shapefile available, or do we trace it?
2. Do we have GEE access (a free non-commercial account works) and someone comfortable with it?
3. Which cooling method do we commit to in section 7.4?
4. Do we include the vulnerability and equity layer, given the time?
5. Do we commit to pymoo NSGA-II on a subset, or reduce the optimization to a greedy plus Pareto sweep if time runs out?
6. Does the guide want Surat only, or is the India-wide story required in the final demo?

---

## 14. Reference list

Method papers
- Voogt, J. A., & Oke, T. R. (2003). Thermal remote sensing of urban climates. Remote Sensing of Environment, 86(3), 370–384.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS. arXiv 1705.07874.
- Lundberg, S. M., et al. (2018). Consistent individualized feature attribution for tree ensembles. arXiv 1802.03888.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. ACM SIGKDD. arXiv 1603.02754.
- Deb, K., et al. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation.

Literature and data pages
- Takacs et al. (2025), npj Urban Sustainability: https://www.nature.com/articles/s42949-025-00271-3 (code and data: https://zenodo.org/records/14041656)
- IRADe Surat heat wave action plan: https://www.apn-gcr.org/wp-content/uploads/2023/06/Climate-Adaptive-and-Gender-Integrated-Heat-Wave-Action-Plan-of-Surat-City.pdf
- Surat and Rajshahi LST maps: https://www.apn-gcr.org/wp-content/uploads/2023/06/Urban-Heat-Island-Maps-of-Surat-and-Rajshahi.pdf
- XGBoost-SHAP hotspot mapping: https://pubmed.ncbi.nlm.nih.gov/40810789/
- XGBoost-SHAP urban form and LST: https://pubmed.ncbi.nlm.nih.gov/40380825/
- Wuhan explainable ML: https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2026.1801455/full
- US cities XAI: https://doi.org/10.3390/su18083694
- Tokyo XGBoost-SHAP: https://www.sciencedirect.com/science/article/abs/pii/S2210670725009485
- LightGBM urban LST: https://pmc.ncbi.nlm.nih.gov/articles/PMC11859931/
- Beijing vacant land NSGA-II: https://www.sciencedirect.com/science/article/abs/pii/S1618866726000944
- Tropical greening optimization: https://www.sciencedirect.com/science/article/abs/pii/S036013232600274X
- Shanghai NSGA-III and XGBoost: https://www.sciencedirect.com/science/article/abs/pii/S0360132325009977
- Blue-green NSGA-II: https://www.nature.com/articles/s41598-026-59100-x
- Tree cooling meta-analysis: https://www.nature.com/articles/s43247-024-01908-4
- Cool and green roofs: https://iopscience.iop.org/article/10.1088/1748-9326/9/5/055002
- Cool pavements meta-analysis: https://www.sciencedirect.com/science/article/pii/S2212095523000640
- Roof mitigation review: https://www.mdpi.com/2075-5309/15/21/3835
- Landsat 8 LST algorithms, Shanghai: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8198215/
- Ahmedabad SUHI: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6749296/
- Surat urbanization and UHI: https://link.springer.com/article/10.1007/s11600-021-00715-1
- USGS Landsat Collection 2 surface temperature: https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature
- Global UHI heat-wave exposure dataset (Scientific Data, 2026): https://www.nature.com/articles/s41597-026-06877-1
- Sen-2 LULC dataset: https://data.mendeley.com/datasets/f4ky6ks248/3

Kaggle
- https://www.kaggle.com/datasets/animeshchoudhury/land-surface-temperature-lst-over-india
- https://www.kaggle.com/datasets/atharvasoundankar/urban-heat-island-uhi-monitoring-dataset
- https://www.kaggle.com/datasets/taruneshburman/urban-heat-island-uhi-effect-prediction
- https://www.kaggle.com/datasets/riccardomogavero/global-urban-heat-island-uhi-data-set
- https://www.kaggle.com/datasets/akheelmd4/extreme-weather-temperature-data-set
