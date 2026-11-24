# The Rain Timetable

This project examines rainfall concentration, continuous low-rain spells, and satellite greenness across years with similar annual rainfall in the southern Great Plains. It compares all eligible year pairs and evaluates ridge regression models using earlier years to predict observations in later years.


## Dataset interpretation

Source: TaoDerong, [Great Plains Multisource NDVI–Climate Time Series (2000–2024)](https://huggingface.co/datasets/TaoDerong/GreatPlains-Multisource-2000-2024). The provider reports MOD09A1-derived regional NDVI, CHIRPS rainfall, and ERA5-Land weather. The original CSV values are unchanged; the analysis selects complete years and omits missing NDVI targets. `data/raw/download_manifest.json` records successful GET checks, timestamps, response sizes, and hashes. The region is approximately 105–95°W and 32–40°N. Regional averaging prevents direct interpretation of low-rain runs as local drought. The analysis uses 2001–2023. 