# Sri Lanka Paddy Intelligence Suite

Streamlit dashboard, ML predictor and natural-language SQL assistant built on official
Department of Census & Statistics paddy data (Maha and Yala seasons, 1979-2026).

## Run

```bash
pip install -r requirements.txt       # add your OpenRouter key (optional, only for the Agri-AI chat)
python -m streamlit run app.py
```

## Data

- `PADDY_DATA_DIR/` holds the source Excel files. `paddy_data.db` is built from them
  (`python database.py` rebuilds it; the app also builds it if the file is missing).
- `paddy_stats`: one row per district / season / year / irrigation type (Major, Minor, Rainfed, Total).
  `province` is filled for every district row (9 provinces); `database.py` adds / back-fills it on start-up
  if an older database lacks it.
  Production is only published for the Total rows. `area_type` separates districts from schemes
  (Udawalawe, Mahaweli H, Highland Paddy) and the national row. Missing source values are NULL.
- `national_paddy_stats`: national season totals 1952-2026 in original units.
- `year` is the calendar year a season ends in (Maha 2023/2024 -> 2024).
- The data has no fertilizer information, so `fertilizer_type` is always NULL.

## Model

`model.py` trains a random forest on district Total rows (the only rows with published production).
Inputs: district, province, season, year, total sown extent and the irrigation mix (Major / Minor /
Rainfed share of the sown extent, plus the dominant irrigation type). The irrigation mix is taken from
the Major / Minor / Rainfed rows of the same district-season; rows without an irrigation split
(before 1984) are not used. Accuracy is measured on the newest 5 years held out (R² ≈ 0.88), and the
model retrains automatically when the database or its feature list changes.

## Dashboard scope

- **Data:** all 25 districts and all years (1979-2025) are loaded. KPI cards and the season pie cover all
  districts for 2015-2025.
- **Year charts** (production trend, year-on-year, district production / yield history, irrigation over time)
  hold every year but open on 2015-2025. Drag sideways to see earlier years.
- **District / province charts** show only the top ones: the province bar and irrigation heatmap show the top 5
  provinces (all districts counted); the scatter plot and yield box plot show the top 10 districts for 2015-2025.
- The District Deep-Dive selector lists all 25 districts; the ML predictor can use any district.
- Years where a season is only partly published (e.g. Maha 1991, Yala 2009) are left out of production totals.
