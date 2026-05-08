# CityMind: An Urban Intelligence System

A Python desktop application that simulates a mid-sized city as a grid-based
graph and runs five integrated AI modules against a single shared graph.
Submitted for the AI2002 Artificial Intelligence semester project at NUCES
Islamabad.

## Group

| Member | Roll No. |
| --- | --- |
| Abdul Rafay | 23i-2027 |
| Kamil Saeed | 23i-2035 |
| Ali Ahmad | 23i-2050 |

## What it does

The application launches a Tkinter GUI containing the city grid, an event log,
overlay toggles, and simulation controls. The five challenge modules cover:

| # | Challenge | Technique |
| - | --------- | --------- |
| 1 | City Layout Planning | CSP: Backtracking + Forward Checking + AC-3 + Min-Conflicts fallback |
| 2 | Road Network Optimization | Kruskal's MST + iterative redundancy with max-flow / edge-disjoint pruning |
| 3 | Ambulance Placement | Genetic Algorithm with Dijkstra-based minimax fitness |
| 4 | Emergency Routing | A\* with Manhattan x 0.8 admissible heuristic, Nearest-Neighbour visit ordering, real-time replanning |
| 5 | Crime Risk Prediction | K-Means clustering -> Random Forest classifier -> feedback into shared graph edge costs |

All five modules read and write the same `CityGraph` object, so any change
(road blocked, risk updated) is immediately visible to every other module.

## How to run

Tkinter is part of the standard Python library on Windows, so no extra GUI
dependency is required. From inside this `citymind/` folder:

```bash
pip install -r requirements.txt
python main.py
```

Or from the folder that *contains* `citymind/`:

```bash
pip install -r citymind/requirements.txt
python -m citymind
```

Either way the same window comes up.

## UI Reference

* **Top bar** — city status, current step, total population, active alerts, weather.
* **Left panel** — navigation, the five Core Modules, system links, system health indicator.
* **Centre** — interactive city grid with overlay toggles (Road Network /
  Ambulance Coverage / Crime Heatmap / All Layers) and a legend strip.
* **Right panel** — Run / Step / Reset, speed selector, live event log,
  active alerts, and a per-cell inspector that fills in when you click a cell.

## Project layout

```
citymind/
  __init__.py
  __main__.py             entry for `python -m citymind`
  main.py                 entry for `python main.py`
  city_graph.py           shared graph - single source of truth
  challenge1_layout.py    CSP layout planner
  challenge2_roads.py     Kruskal MST + max-flow edge-disjoint redundancy
  challenge3_ambulance.py Genetic Algorithm placement
  challenge4_routing.py   A* dynamic routing
  challenge5_crime.py     K-Means + Random Forest pipeline
  simulation.py           20-step simulation orchestrator
  ui/
    app.py                main Tk window
    city_grid.py          grid canvas with overlays
    event_log.py          event log panel
    theme.py              colour and font palette
  requirements.txt
  README.md
```

## Defending the design in viva

* The shared `CityGraph` object is created once in `simulation.build_simulation`
  and passed by reference everywhere. There are no copies. Trace any single
  edge block from `simulation._block_random_edges` -> `EdgeData.blocked`
  -> `EmergencyRouter.a_star` to confirm the propagation.
* Every algorithm choice in the design document is implemented as written -
  see the `Why not alternatives` notes in the design PDF for the
  justifications.
* Each module exposes a clean function (`plan_layout`, `build_road_network`,
  `place_ambulances`, `predict_crime_risk`, `EmergencyRouter.a_star`) so the
  simulation loop never reaches into algorithm internals.
