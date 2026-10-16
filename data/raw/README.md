# data/raw

Empty on purpose. Put downloaded datasets here, for example:

- Kaggle: Land Surface Temperature (LST) over India
  https://www.kaggle.com/datasets/animeshchoudhury/land-surface-temperature-lst-over-india
- Any Google Earth Engine exports (Landsat 8/9 Collection 2 Level 2, Sentinel-2 SR,
  Dynamic World, Open Buildings, OSM extracts) — export as GeoTIFF/GeoJSON/CSV into
  this folder, keeping one subfolder per source, e.g. raw/landsat, raw/sentinel2,
  raw/osm, raw/open_buildings.

Nothing in this folder is tracked by git except this file and .gitkeep.
See ml/src/data/ for the scripts that read from here.
