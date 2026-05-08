"""Main CityMind application window.

Lays out the wireframe faithfully:
    - Top bar (status strip)
    - Left navigation panel
    - Centre city map with overlay toggles + legend
    - Right controls + event log + active alerts
DPI-aware on Windows, ttk-styled throughout for crisp text and consistent
hover behaviour.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, List, Optional, Set, Tuple

from . import theme
from .city_grid import CityGridCanvas
from .event_log import EventLogPanel
from .widgets import IconButton, PillToggle
from ..city_graph import (
    CityGraph,
    LETTER_CODE,
    LOC_AMBULANCE_DEPOT,
    LOC_HOSPITAL,
    LOC_INDUSTRIAL,
    LOC_POWER_PLANT,
    LOC_RESIDENTIAL,
    LOC_SCHOOL,
)
from ..simulation import (
    CitySimulation,
    SimulationEvent,
    TOTAL_STEPS,
    build_simulation,
)


SPEED_DELAYS = {"0.5x": 1600, "1x": 800, "2x": 400, "4x": 200}
DEFAULT_SPEED = "1x"
DEFAULT_GRID = 8
WINDOW_W = 1440
WINDOW_H = 900
LEFT_NAV_W = 240
RIGHT_PANEL_W = 360


class CityMindApp(tk.Tk):
    """Top-level Tk window."""

    def __init__(self):
        theme.enable_dpi_awareness()
        super().__init__()
        self.title("CityMind - Urban Intelligence System")
        self.configure(bg=theme.BG_BASE)
        theme.apply_tk_scaling(self)
        theme.setup_fonts(self)
        theme.setup_styles(self)

        self.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.minsize(1180, 760)

        self.graph: Optional[CityGraph] = None
        self.simulation: Optional[CitySimulation] = None
        self._auto_running = False
        self._speed = DEFAULT_SPEED
        self._after_id: Optional[str] = None
        self._nav_buttons: Dict[str, IconButton] = {}
        self._toggle_widgets: Dict[str, PillToggle] = {}

        self._build_layout()
        self._reset()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        # Three-column grid. Top bar spans all columns.
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self._build_top_bar()
        self._build_left_nav()
        self._build_centre()
        self._build_right_panel()

    # ------------------------------- top bar -----------------------------

    def _build_top_bar(self) -> None:
        bar = ttk.Frame(self, style="Panel.TFrame", height=64)
        bar.grid(row=0, column=0, columnspan=3, sticky="ew")
        bar.grid_propagate(False)

        # Visual divider under the bar.
        ttk.Separator(self, orient="horizontal").grid(row=0, column=0,
                                                        columnspan=3,
                                                        sticky="sew")

        labels = ["City Status", "Simulation", "Population", "Active Alerts",
                  "Weather"]
        defaults = ["Idle", f"Step 0 of {TOTAL_STEPS}", "-", "0", "Clear"]

        self._status_vars: Dict[str, tk.StringVar] = {
            label: tk.StringVar(value=value)
            for label, value in zip(labels, defaults)
        }

        for i, label in enumerate(labels):
            cell = ttk.Frame(bar, style="Panel.TFrame", padding=(20, 12))
            cell.grid(row=0, column=i, sticky="nsw")
            ttk.Label(cell, text=label.upper(),
                      style="Caption.TLabel").pack(anchor="w")
            ttk.Label(cell, textvariable=self._status_vars[label],
                      style="Status.TLabel").pack(anchor="w", pady=(2, 0))
            if i < len(labels) - 1:
                sep = tk.Frame(bar, bg=theme.BG_PANEL_DEEP, width=1)
                sep.grid(row=0, column=i, sticky="nse")

    # ------------------------------- left nav ----------------------------

    def _build_left_nav(self) -> None:
        nav = ttk.Frame(self, style="Panel.TFrame", width=LEFT_NAV_W)
        nav.grid(row=1, column=0, sticky="ns")
        nav.grid_propagate(False)

        # Brand block.
        brand = ttk.Frame(nav, style="Panel.TFrame", padding=(20, 22, 20, 16))
        brand.pack(fill="x")
        ttk.Label(brand, text="CityMind", style="Brand.TLabel").pack(anchor="w")
        ttk.Label(brand, text="Urban Intelligence System",
                  style="BrandSub.TLabel").pack(anchor="w")

        # Sections.
        self._build_nav_section(nav, "Navigation", [
            ("Dashboard", lambda: self._show_module(0)),
            ("City Overview", lambda: self._show_module(-1)),
        ])
        self._build_nav_section(nav, "Core Modules", [
            ("1. City Layout Planning", lambda: self._show_module(1)),
            ("2. Road Network Optimization", lambda: self._show_module(2)),
            ("3. Ambulance Placement", lambda: self._show_module(3)),
            ("4. Emergency Routing", lambda: self._show_module(4)),
            ("5. Crime Risk Prediction", lambda: self._show_module(5)),
        ])
        self._build_nav_section(nav, "System", [
            ("Event Log", lambda: self._set_info("Event log on right panel.")),
            ("Settings", lambda: self._set_info("Settings: dark theme only.")),
        ])

        spacer = ttk.Frame(nav, style="Panel.TFrame")
        spacer.pack(fill="both", expand=True)

        health = ttk.Frame(nav, style="PanelAlt.TFrame",
                            padding=(14, 12))
        health.pack(fill="x", side="bottom", padx=14, pady=14)
        ttk.Label(health, text="SYSTEM HEALTH",
                  style="Caption.PanelAlt.TLabel").pack(anchor="w")
        self._health_var = tk.StringVar(value="● Nominal")
        ttk.Label(health, textvariable=self._health_var,
                  style="PanelAlt.TLabel",
                  foreground=theme.GOOD,
                  font=theme.FONT_BODY_BOLD).pack(anchor="w", pady=(4, 0))

    def _build_nav_section(self, parent, title: str,
                           buttons: List[Tuple[str, Callable]]) -> None:
        section = ttk.Frame(parent, style="Panel.TFrame",
                              padding=(14, 8, 14, 8))
        section.pack(fill="x")
        ttk.Label(section, text=title.upper(),
                  style="Caption.TLabel").pack(anchor="w", pady=(2, 8))
        for label, command in buttons:
            btn = IconButton(section, label=label,
                              command=lambda l=label, c=command: self._handle_nav(l, c))
            btn.pack(fill="x", pady=1)
            self._nav_buttons[label] = btn

    def _handle_nav(self, label: str, command: Callable) -> None:
        for btn_label, btn in self._nav_buttons.items():
            btn.set_active(btn_label == label)
        command()

    # ------------------------------- centre ------------------------------

    def _build_centre(self) -> None:
        centre = ttk.Frame(self, style="App.TFrame")
        centre.grid(row=1, column=1, sticky="nsew")
        centre.grid_rowconfigure(2, weight=1)
        centre.grid_columnconfigure(0, weight=1)

        title = ttk.Frame(centre, style="App.TFrame", padding=(24, 18, 24, 6))
        title.grid(row=0, column=0, sticky="ew")
        ttk.Label(title, text="CITY MAP", style="Title.TLabel").pack(anchor="w")
        ttk.Label(title, text="Interactive Grid View",
                  style="Heading.TLabel",
                  foreground=theme.TEXT_DIM).pack(anchor="w", pady=(2, 0))

        # Overlay toggles.
        toggles = ttk.Frame(centre, style="App.TFrame", padding=(24, 4, 24, 8))
        toggles.grid(row=1, column=0, sticky="ew")

        for label, key in [("Road Network", "roads"),
                            ("Ambulance Coverage", "coverage"),
                            ("Crime Heatmap", "heatmap")]:
            t = PillToggle(toggles, label=label, initial=True,
                            on_toggle=lambda v, k=key: self._on_toggle(k, v))
            t.pack(side="left", padx=(0, 10))
            self._toggle_widgets[key] = t

        # "All Layers" - acts as a master toggle.
        all_layers = PillToggle(toggles, label="All Layers", initial=True,
                                  on_toggle=self._on_all_layers,
                                  width=130)
        all_layers.pack(side="left", padx=(8, 0))
        self._toggle_widgets["all"] = all_layers

        # Canvas card.
        canvas_card = ttk.Frame(centre, style="Card.TFrame")
        canvas_card.grid(row=2, column=0, sticky="nsew", padx=24, pady=(2, 12))
        self.grid_canvas = CityGridCanvas(
            canvas_card, graph=self._fresh_graph(),
            on_cell_click=self._on_cell_click)
        self.grid_canvas.pack(fill="both", expand=True, padx=10, pady=10)

        # Legend.
        legend = ttk.Frame(centre, style="App.TFrame", padding=(24, 4, 24, 16))
        legend.grid(row=3, column=0, sticky="ew")
        for loc_type in (LOC_RESIDENTIAL, LOC_HOSPITAL, LOC_SCHOOL,
                         LOC_INDUSTRIAL, LOC_POWER_PLANT, LOC_AMBULANCE_DEPOT):
            self._legend_chip(legend, LETTER_CODE[loc_type], loc_type,
                                colour=theme.TYPE_BORDER[loc_type]).pack(
                                    side="left", padx=(0, 16))
        self._legend_chip(legend, "—", "Road",
                            colour=theme.ACCENT_DIM).pack(side="left",
                                                            padx=(0, 16))
        self._legend_chip(legend, "---", "Blocked Road",
                            colour=theme.DANGER).pack(side="left")

    def _legend_chip(self, parent, code: str, name: str,
                      colour: str) -> tk.Frame:
        frame = tk.Frame(parent, bg=theme.BG_BASE)
        chip = tk.Label(frame, text=code, bg=theme.BG_PANEL_ALT,
                          fg=colour, font=theme.FONT_BODY_BOLD,
                          padx=10, pady=2, width=4)
        chip.pack(side="left", padx=(0, 6))
        tk.Label(frame, text=name, bg=theme.BG_BASE,
                  fg=theme.TEXT_DIM, font=theme.FONT_SMALL).pack(side="left")
        return frame

    # ------------------------------- right panel -------------------------

    def _build_right_panel(self) -> None:
        right = ttk.Frame(self, style="Panel.TFrame", width=RIGHT_PANEL_W)
        right.grid(row=1, column=2, sticky="ns")
        right.grid_propagate(False)
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # Header + controls.
        controls_card = ttk.Frame(right, style="Panel.TFrame",
                                    padding=(18, 18, 18, 8))
        controls_card.grid(row=0, column=0, sticky="ew")
        ttk.Label(controls_card, text="SIMULATION CONTROLS",
                  style="Caption.TLabel").pack(anchor="w", pady=(0, 10))

        self._run_btn = ttk.Button(controls_card, text="▶  Run Simulation",
                                     style="Primary.TButton",
                                     command=self._on_run)
        self._run_btn.pack(fill="x", pady=(0, 8))

        self._step_btn = ttk.Button(controls_card, text="⏭  Step Forward",
                                      style="Secondary.TButton",
                                      command=self._on_step)
        self._step_btn.pack(fill="x", pady=(0, 8))

        self._reset_btn = ttk.Button(controls_card, text="↻  Reset",
                                       style="Secondary.TButton",
                                       command=self._reset)
        self._reset_btn.pack(fill="x", pady=(0, 4))

        # Speed selector.
        speed_card = ttk.Frame(right, style="Panel.TFrame",
                                 padding=(18, 8, 18, 14))
        speed_card.grid(row=1, column=0, sticky="ew")
        ttk.Label(speed_card, text="SPEED",
                  style="Caption.TLabel").pack(anchor="w", pady=(0, 6))
        speed_row = ttk.Frame(speed_card, style="Panel.TFrame")
        speed_row.pack(fill="x")
        speed_row.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self._speed_buttons: Dict[str, ttk.Button] = {}
        for i, label in enumerate(SPEED_DELAYS):
            btn = ttk.Button(speed_row, text=label, style="Speed.TButton",
                              command=lambda l=label: self._set_speed(l))
            btn.grid(row=0, column=i, padx=2, sticky="ew")
            self._speed_buttons[label] = btn
        self._highlight_speed(DEFAULT_SPEED)

        # Event log.
        self.event_log = EventLogPanel(right)
        self.event_log.grid(row=2, column=0, sticky="nsew", padx=14, pady=8)

        # Active alerts.
        alerts = ttk.Frame(right, style="PanelAlt.TFrame",
                            padding=(16, 12, 16, 12))
        alerts.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 8))
        ttk.Label(alerts, text="ACTIVE ALERTS",
                  style="Caption.PanelAlt.TLabel").pack(anchor="w",
                                                         pady=(0, 4))
        self._alert_var = tk.StringVar(value="No active alerts.")
        ttk.Label(alerts, textvariable=self._alert_var,
                  style="PanelAlt.TLabel",
                  foreground=theme.ALERT,
                  font=theme.FONT_SMALL,
                  wraplength=300, justify="left").pack(anchor="w", fill="x")

        # Inspector.
        inspector = ttk.Frame(right, style="Panel.TFrame",
                                padding=(16, 12, 16, 14))
        inspector.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 14))
        ttk.Label(inspector, text="INSPECTOR",
                  style="Caption.TLabel").pack(anchor="w", pady=(0, 4))
        self._inspector_var = tk.StringVar(value="Click any cell to inspect.")
        ttk.Label(inspector, textvariable=self._inspector_var,
                  style="Inspector.TLabel",
                  wraplength=300, justify="left").pack(anchor="w", fill="x")

    # ------------------------------------------------------------------
    # Sim lifecycle
    # ------------------------------------------------------------------

    def _fresh_graph(self) -> CityGraph:
        graph, sim = build_simulation(rows=DEFAULT_GRID, cols=DEFAULT_GRID)
        self.graph = graph
        self.simulation = sim
        return graph

    def _reset(self) -> None:
        self._cancel_auto()
        graph, sim = build_simulation(rows=DEFAULT_GRID, cols=DEFAULT_GRID,
                                       seed=None)
        self.graph = graph
        self.simulation = sim
        sim.initialise()
        sim.on_event(self._on_event)

        self.event_log.clear()
        for event in sim.state.events:
            self.event_log.add_event(event)

        self.grid_canvas.graph = graph
        self._sync_canvas_state()
        self._refresh_status_bar()

    def _on_event(self, event: SimulationEvent) -> None:
        # Marshal to Tk thread to avoid race conditions.
        self.after(0, lambda: self._handle_event(event))

    def _handle_event(self, event: SimulationEvent) -> None:
        self.event_log.add_event(event)
        if event.kind in {"flood", "warning"}:
            self._alert_var.set(event.message)

    def _on_run(self) -> None:
        if self.simulation is None or self.simulation.state.finished:
            return
        self._auto_running = not self._auto_running
        self._run_btn.configure(text="❚❚  Pause" if self._auto_running
                                else "▶  Run Simulation")
        if self._auto_running:
            self._tick()

    def _tick(self) -> None:
        if not self._auto_running or self.simulation is None:
            return
        if not self.simulation.state.finished:
            self.simulation.step()
            self._sync_canvas_state()
            self._refresh_status_bar()
        if self.simulation.state.finished:
            self._auto_running = False
            self._run_btn.configure(text="▶  Run Simulation")
            return
        delay = SPEED_DELAYS[self._speed]
        self._after_id = self.after(delay, self._tick)

    def _on_step(self) -> None:
        if self.simulation is None or self.simulation.state.finished:
            return
        self.simulation.step()
        self._sync_canvas_state()
        self._refresh_status_bar()

    def _cancel_auto(self) -> None:
        self._auto_running = False
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if hasattr(self, "_run_btn"):
            self._run_btn.configure(text="▶  Run Simulation")

    def _set_speed(self, label: str) -> None:
        self._speed = label
        self._highlight_speed(label)

    def _highlight_speed(self, label: str) -> None:
        for name, btn in self._speed_buttons.items():
            btn.configure(style="SpeedActive.TButton" if name == label
                          else "Speed.TButton")

    # ------------------------------------------------------------------
    # Toggles
    # ------------------------------------------------------------------

    def _on_toggle(self, key: str, value: bool) -> None:
        self.grid_canvas.set_overlay(key, value)
        # Sync the master "All Layers" pill to reflect the combined state.
        master = self._toggle_widgets.get("all")
        if master is not None:
            individual = ("roads", "coverage", "heatmap")
            master.set(all(self._toggle_widgets[k].get() for k in individual))

    def _on_all_layers(self, value: bool) -> None:
        for k in ("roads", "coverage", "heatmap"):
            self._toggle_widgets[k].set(value)
        self.grid_canvas.set_overlay("all", value)

    # ------------------------------------------------------------------
    # Canvas / status sync
    # ------------------------------------------------------------------

    def _sync_canvas_state(self) -> None:
        if self.simulation is None:
            return
        state = self.simulation.state
        ambulances = state.ambulances.positions if state.ambulances else []
        police = state.crime.police_deployment if state.crime else []
        self.grid_canvas.set_simulation_state(
            ambulances=ambulances,
            civilians=state.civilians,
            medical_team_position=state.medical_team_position,
            police=police,
            recently_blocked=self._recently_blocked_edges(),
            current_route=self._current_route(),
        )

    def _recently_blocked_edges(self) -> Set[Tuple[int, int]]:
        if self.simulation is None or self.graph is None:
            return set()
        current_step = self.simulation.state.step
        return {
            (u, v) for u, v, edge in self.graph.all_edges()
            if edge.flash_until_step >= current_step and edge.blocked
        }

    def _current_route(self) -> List[int]:
        if (self.simulation is None or self.simulation.state.finished
                or not self.simulation.state.civilians
                or self.simulation.state.medical_team_position is None):
            return []
        path, _ = self.simulation.router.a_star(
            self.simulation.state.medical_team_position,
            self.simulation.state.civilians[0])
        return path

    def _refresh_status_bar(self) -> None:
        if self.simulation is None or self.graph is None:
            return
        state = self.simulation.state
        self._status_vars["Simulation"].set(
            f"Step {state.step} of {TOTAL_STEPS}")
        total_pop = sum(n.population_density for n in self.graph.all_nodes())
        self._status_vars["Population"].set(f"{total_pop:,}")
        if state.finished:
            self._status_vars["City Status"].set("Complete")
        elif self._auto_running:
            self._status_vars["City Status"].set("Running")
        else:
            self._status_vars["City Status"].set("Ready")

        active_alerts = sum(1 for _, _, edge in self.graph.all_edges()
                            if edge.blocked)
        self._status_vars["Active Alerts"].set(str(active_alerts))
        if active_alerts > 4:
            weather = "Stormy"
        elif active_alerts > 1:
            weather = "Light Rain"
        else:
            weather = "Clear"
        self._status_vars["Weather"].set(weather)

        if active_alerts == 0:
            self._alert_var.set("No active alerts.")
        else:
            self._alert_var.set(f"{active_alerts} road(s) flooded. "
                                f"Dynamic router replanning.")

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _show_module(self, idx: int) -> None:
        descriptions = {
            0: "Dashboard - overall system view.",
            -1: "City overview - aggregate counts and population stats.",
            1: "Challenge 1: CSP layout. Industrial != adjacent to school/hospital. "
               "Residential within 3 hops of hospital. Power plant within 2 hops "
               "of industrial.",
            2: "Challenge 2: Kruskal MST + max-flow >= 2 between Primary Hospital "
               "and Ambulance Depot.",
            3: "Challenge 3: GA places 3 ambulances. Fitness = max Dijkstra distance "
               "from any residential to nearest ambulance.",
            4: "Challenge 4: A* with Manhattan x 0.8 admissible heuristic. "
               "Real-time replanning when roads block.",
            5: "Challenge 5: K-Means -> Random Forest. Risk levels feed back into "
               "edge effective_cost via 1.0 / 1.2 / 1.5 multipliers.",
        }
        self._inspector_var.set(descriptions.get(idx, ""))

    def _set_info(self, message: str) -> None:
        self._inspector_var.set(message)

    def _on_cell_click(self, node_id: int) -> None:
        if self.graph is None:
            return
        n = self.graph.node(node_id)
        is_amb = (self.simulation
                  and self.simulation.state.ambulances
                  and node_id in self.simulation.state.ambulances.positions)
        details = (
            f"Cell #{n.id} ({n.row}, {n.col})\n"
            f"Type: {n.type}\n"
            f"Population: {n.population_density}\n"
            f"Crime risk: {n.crime_risk_level}\n"
            f"Primary hospital: {'Yes' if n.is_primary_hospital else 'No'}\n"
            f"Ambulance posted: {'Yes' if is_amb else 'No'}"
        )
        self._inspector_var.set(details)


def run_app() -> None:
    """Entry point called by main.py."""
    app = CityMindApp()
    app.mainloop()
