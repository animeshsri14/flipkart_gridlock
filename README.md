# Gridlock — Smart Traffic Management for VIP Motorcades

Built for the **Flipkart Grid 6.0 Robotics Challenge** (Smart Traffic Management problem statement).

Gridlock helps Bengaluru Traffic Police plan and manage road closures during VIP motorcade events. It predicts which corridors get hit, recommends where to place barricades and police postings, and generates bypass routes — all from a single Streamlit dashboard.

---

## What it does

1. **Forecasts corridor impact** — takes historical event data (Astram dataset), runs time-series analysis, and predicts severity + duration of disruptions per corridor.
2. **Scores barricade & posting placements** — a multi-factor heuristic engine ranks deployment options by road width, intersection density, pedestrian volume, and incident history.
3. **Generates bypass routes** — uses OSRM to compute real driving routes, with perpendicular offsets to find clean detours around blocked corridors.
4. **Learns from feedback** — officers rate deployment effectiveness after events; that data feeds back into future recommendations.

## Corridors covered

| Corridor | Anchor (lat, lon) | Bearing |
|---|---|---|
| Mysore Road | 12.945, 77.530 | 110° |
| Bellary Road 1 | 13.030, 77.590 | 10° |
| Tumkur Road | 13.020, 77.530 | 300° |
| Hosur Road | 12.900, 77.620 | 150° |

---

## Project structure

```
├── app.py              # Streamlit dashboard — map viz, sidebar controls, deployment plans
├── data_pipeline.py    # CSV ingestion, cleaning, feature engineering (time-based)
├── forecaster.py       # Exponential smoothing + weighted moving averages per corridor
├── predictions.py      # Combines forecaster + heuristics into deployment recommendations
├── heuristics.py       # Multi-factor scoring engine for barricade/posting placement
├── geo.py              # Haversine, bearing, destination point — the spatial math
├── feedback_store.py   # SQLite CRUD for officer feedback on past deployments
├── test_smoke.py       # Pytest smoke tests across all modules
├── BLUEPRINT.md        # Detailed design document and architecture notes
└── .gitignore
```

### How the pieces fit together

```
app.py (UI + orchestration)
 ├── data_pipeline    → loads CSV, computes time features
 ├── forecaster       → predicts severity/duration per corridor
 ├── heuristics       → scores and ranks deployment options
 ├── predictions      → fuses forecast + heuristics into recommendations
 ├── geo              → coordinate math for bypass route offsets
 ├── feedback_store   → persists officer ratings (SQLite)
 └── OSRM API         → fetches real driving routes for bypasses
```

## Setup

**Requirements:** Python 3.9+

```bash
pip install streamlit pydeck pandas numpy
```

**Run:**

```bash
streamlit run app.py
```

The dashboard opens at `localhost:8501`. Use the sidebar to pick a corridor, set time windows, and adjust confidence thresholds.

**Tests:**

```bash
pytest test_smoke.py -v
```

## External services

- [OSRM](https://router.project-osrm.org) — public routing API, no key needed. Used for bypass route geometry.

## Data

The Astram event CSV (~4.5 MB) and SQLite feedback database are excluded from version control. Place the CSV in the project root to use the pipeline.

---

*Flipkart Grid 6.0 — Robotics Challenge submission*
