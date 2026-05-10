"""20-step CityMind simulation orchestrator.

This module wires the five challenges together against the shared CityGraph.
The UI calls into here for each step; the simulation never owns its own copy
of any module's results - every change flows through the graph.

Per-step behaviour (matches design doc):
 1. Block 0-2 random edges to simulate flooding events.
 2. Run dynamic A* re-planning for the medical team if its path is hit.
 3. If ambulance coverage degrades significantly, re-run the GA.
 4. Every CRIME_REFRESH_INTERVAL steps, re-run the ML pipeline so risk
    multipliers respond to the current environment.
 5. Append a human-readable entry to the event log for every action taken.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from .challenge1_layout import LayoutResult, plan_layout
from .challenge2_roads import RoadNetworkResult, build_road_network
from .challenge3_ambulance import AmbulanceResult, place_ambulances
from .challenge4_routing import EmergencyRouter, RoutingResult
from .challenge5_crime import CrimeResult, predict_crime_risk
from .city_graph import CityGraph, LOC_EMPTY, LOC_RESIDENTIAL


TOTAL_STEPS = 20
MAX_BLOCKS_PER_STEP = 2
COVERAGE_DEGRADATION_THRESHOLD = 1.5  # how much worst-case can grow before GA reruns
CRIME_REFRESH_INTERVAL = 5            # re-run ML every N steps
DEFAULT_GRID_SIZE = 8


@dataclass
class SimulationEvent:
    """One log entry produced by the simulation."""

    step: int
    kind: str
    message: str


@dataclass
class SimulationState:
    """Live snapshot of the simulation - exposed to the UI for rendering."""

    step: int = 0
    layout: Optional[LayoutResult] = None
    roads: Optional[RoadNetworkResult] = None
    crime: Optional[CrimeResult] = None
    ambulances: Optional[AmbulanceResult] = None
    last_route: Optional[RoutingResult] = None
    civilians: List[int] = field(default_factory=list)
    medical_team_position: Optional[int] = None
    events: List[SimulationEvent] = field(default_factory=list)
    finished: bool = False


class CitySimulation:
    """Drives the 20-step simulation against the shared CityGraph."""

    def __init__(self, graph: CityGraph, seed: Optional[int] = None):
        self.graph = graph
        self.seed = seed
        self.rng = random.Random(seed)
        self.state = SimulationState()
        self.router = EmergencyRouter(graph)
        self._listeners: List[Callable[[SimulationEvent], None]] = []
        self._baseline_worst_case: float = math.inf

    # ------------------------------------------------------------------
    # Listener registration so the UI can react to events live.
    # ------------------------------------------------------------------

    def on_event(self, listener: Callable[[SimulationEvent], None]) -> None:
        self._listeners.append(listener)

    def _emit(self, kind: str, message: str) -> None:
        event = SimulationEvent(step=self.state.step, kind=kind, message=message)
        self.state.events.append(event)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as exc:
                print(f"[simulation] listener failed: {exc}")

    # ------------------------------------------------------------------
    # Initialisation - run the four "design-time" modules in order.
    # ------------------------------------------------------------------

    def initialise(self) -> None:
        self.state.step = 0
        self._emit("init", "City graph created. Starting initial pipeline.")

        layout = plan_layout(self.graph, seed=self.seed)
        self.state.layout = layout
        if layout.success and not layout.used_min_conflicts:
            self._emit("layout", "CSP backtracking solved layout cleanly.")
        elif layout.success and layout.used_min_conflicts:
            self._emit("layout",
                       f"CSP fell back to Min-Conflicts; converged to a valid "
                       f"layout in {layout.iterations} steps.")
        else:
            self._emit("layout",
                       f"Min-Conflicts could not eliminate every violation. "
                       f"Most-violated rule: {layout.violated_rule}.")

        roads = build_road_network(self.graph)
        self.state.roads = roads
        self._emit("roads",
                   f"Road network built. cost={roads.total_cost:.2f}, "
                   f"hospital<->depot edge-disjoint paths={roads.edge_disjoint_paths}, "
                   f"redundancy edges added={roads.extra_edges_added}.")

        crime = predict_crime_risk(self.graph, seed=self.seed)
        self.state.crime = crime
        high = sum(1 for v in crime.risk_levels.values() if v == "High")
        med = sum(1 for v in crime.risk_levels.values() if v == "Medium")
        low = sum(1 for v in crime.risk_levels.values() if v == "Low")
        self._emit("crime",
                   f"Crime ML pipeline complete (RF train acc={crime.accuracy:.2f}). "
                   f"High={high}, Medium={med}, Low={low}. Police deployed at "
                   f"{len(crime.police_deployment)} locations.")

        ambulances = place_ambulances(self.graph, seed=self.seed)
        self.state.ambulances = ambulances
        self._baseline_worst_case = ambulances.worst_case_distance
        self._emit("ambulance",
                   f"Ambulance GA placed at {ambulances.positions} "
                   f"(worst-case={ambulances.worst_case_distance:.2f}, "
                   f"generations={ambulances.generations_run}).")

        # Pick three random residential cells to act as trapped civilians and
        # stage the medical team at the depot for the simulation steps.
        residentials = [n.id for n in self.graph.all_nodes()
                        if n.type == LOC_RESIDENTIAL]
        self.rng.shuffle(residentials)
        self.state.civilians = residentials[:3]
        depot_id = self.graph.ambulance_depot_id
        if depot_id is None and ambulances.positions:
            depot_id = ambulances.positions[0]
        self.state.medical_team_position = depot_id
        self._emit("mission",
                   f"Medical team staged at node {depot_id}. "
                   f"Targets: {self.state.civilians}.")

    # ------------------------------------------------------------------
    # Per-step behaviour.
    # ------------------------------------------------------------------

    def step(self) -> bool:
        """Advance the simulation by one step. Returns True if more steps remain."""
        if self.state.finished:
            return False

        self.state.step += 1
        step = self.state.step

        # 1. Random edge blocks for this step.
        blocked_now = self._block_random_edges(step)

        # 2. Run/continue the medical-team route. We replan from current
        #    position toward the next civilian on every step, which gives the
        #    UI a moving target without complicating the algorithm.
        self._advance_medical_team()

        # 3. Re-evaluate ambulance coverage if the graph has changed materially.
        if blocked_now and self._coverage_degraded():
            new_amb = place_ambulances(self.graph, seed=self.seed + step
                                       if self.seed is not None else None)
            self.state.ambulances = new_amb
            self._baseline_worst_case = new_amb.worst_case_distance
            self._emit("ambulance",
                       f"GA re-ran after road block. New worst-case="
                       f"{new_amb.worst_case_distance:.2f}.")

        # 4. Periodic crime-risk refresh - keeps the heatmap honest as the
        #    population/road state shifts.
        if step % CRIME_REFRESH_INTERVAL == 0:
            crime = predict_crime_risk(self.graph,
                                       seed=(self.seed + step
                                             if self.seed is not None else None))
            self.state.crime = crime
            self._emit("crime",
                       f"Crime ML pipeline re-ran. RF train acc={crime.accuracy:.2f}.")

        if step >= TOTAL_STEPS:
            self.state.finished = True
            self._emit("finish", "20-step simulation complete.")
        return not self.state.finished

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    def _block_random_edges(self, step: int) -> bool:
        """Block 0-2 random unblocked, in-use edges this step, and clear old blocks."""
        # Clear floods older than 3 steps
        for u, v, edge in self.graph.all_edges():
            if edge.blocked and getattr(edge, "flash_until_step", 0) < step - 2:
                self.graph.unblock_edge(u, v)

        candidates = [
            (u, v) for u, v, edge in self.graph.all_edges()
            if not edge.blocked
            and self.graph.node(u).type != LOC_EMPTY
            and self.graph.node(v).type != LOC_EMPTY
        ]
        if not candidates:
            return False
        count = self.rng.randint(0, MAX_BLOCKS_PER_STEP)
        if count == 0:
            return False
        blocked: List[Tuple[int, int]] = []
        self.rng.shuffle(candidates)
        for u, v in candidates[:count]:
            self.graph.block_edge(u, v, current_step=step)
            blocked.append((u, v))
        if blocked:
            description = ", ".join(f"({u}-{v})" for u, v in blocked)
            self._emit("flood", f"Roads flooded: {description}.")
        return bool(blocked)

    def _advance_medical_team(self) -> None:
        """Re-plan from current team position; advance one A* hop along the path."""
        position = self.state.medical_team_position
        if position is None or not self.state.civilians:
            return

        target = self.state.civilians[0]
        path, cost = self.router.a_star(position, target)
        if not path or cost == math.inf:
            self._emit("route",
                       f"Civilian {target} unreachable - waiting for floods to clear.")
            return

        if len(path) <= 1:
            # Already at the target.
            self._emit("rescue",
                       f"Civilian rescued at node {target}. "
                       f"Remaining: {self.state.civilians[1:]}.")
            self.state.civilians.pop(0)
            return

        next_node = path[1]
        edge = self.graph.edge(position, next_node)
        if edge.blocked:
            # Re-plan was needed - try a different path. (a_star already
            # returned a clear path, so this branch fires only on race conditions.)
            self._emit("route",
                       f"Replanning - leg ({position}->{next_node}) is blocked.")
            return

        self.state.medical_team_position = next_node
        self._emit("route",
                   f"Medical team {position}->{next_node} "
                   f"(towards civilian {target}).")
        if next_node == target:
            self._emit("rescue",
                       f"Civilian rescued at node {target}. "
                       f"Remaining: {self.state.civilians[1:]}.")
            self.state.civilians.pop(0)

    def _coverage_degraded(self) -> bool:
        """Check whether the ambulance worst-case has worsened enough to re-run GA."""
        if self.state.ambulances is None:
            return False
        weighted_router = EmergencyRouter(self.graph)
        worst = 0.0
        for n in self.graph.all_nodes():
            if n.type != LOC_RESIDENTIAL:
                continue
            best_for_resident = math.inf
            for amb in self.state.ambulances.positions:
                _, cost = weighted_router.a_star(amb, n.id)
                if cost < best_for_resident:
                    best_for_resident = cost
            if best_for_resident > worst:
                worst = best_for_resident

        if self._baseline_worst_case == math.inf:
            self._baseline_worst_case = worst
            return False
        return worst > self._baseline_worst_case * COVERAGE_DEGRADATION_THRESHOLD


def build_simulation(rows: int = DEFAULT_GRID_SIZE,
                     cols: int = DEFAULT_GRID_SIZE,
                     seed: Optional[int] = None) -> Tuple[CityGraph, CitySimulation]:
    """Factory that produces a ready-to-run simulation."""
    graph = CityGraph(rows=rows, cols=cols)
    sim = CitySimulation(graph, seed=seed)
    return graph, sim
