"""Shared CityGraph - the single source of truth for the entire CityMind system.

Every challenge module operates on the same CityGraph instance by reference. There
are no copies. A write to a node or edge here is immediately visible to every
reader without any synchronization layer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx


# Location type tags. Single-letter codes are used in the UI grid.
LOC_RESIDENTIAL = "Residential"
LOC_HOSPITAL = "Hospital"
LOC_SCHOOL = "School"
LOC_INDUSTRIAL = "Industrial"
LOC_POWER_PLANT = "Power Plant"
LOC_AMBULANCE_DEPOT = "Ambulance Depot"
LOC_EMPTY = "Empty"

LOCATION_TYPES = (
    LOC_RESIDENTIAL,
    LOC_HOSPITAL,
    LOC_SCHOOL,
    LOC_INDUSTRIAL,
    LOC_POWER_PLANT,
    LOC_AMBULANCE_DEPOT,
)

LETTER_CODE = {
    LOC_RESIDENTIAL: "R",
    LOC_HOSPITAL: "H",
    LOC_SCHOOL: "S",
    LOC_INDUSTRIAL: "I",
    LOC_POWER_PLANT: "P",
    LOC_AMBULANCE_DEPOT: "A",
    LOC_EMPTY: ".",
}

# Risk-to-cost multipliers from the design document.
RISK_MULTIPLIER = {"Low": 1.0, "Medium": 1.2, "High": 1.5}

# Edge base costs from the project specification.
COST_STANDARD = 1.0
COST_RESIDENTIAL = 0.8


@dataclass
class NodeData:
    """Properties stored on every node in the city graph."""

    id: int
    row: int
    col: int
    type: str = LOC_EMPTY
    population_density: int = 0
    risk_index: float = 0.0
    accessibility: bool = True
    crime_risk_level: str = "Low"
    is_primary_hospital: bool = False


@dataclass
class EdgeData:
    """Properties stored on every edge in the city graph."""

    base_cost: float = COST_STANDARD
    blocked: bool = False
    effective_cost: float = COST_STANDARD
    flash_until_step: int = -1  # used by the UI to flash red when newly blocked


class CityGraph:
    """The shared city graph. One instance is created at startup.

    The graph is laid out as a rows x cols grid. Each cell is a node connected
    to its 4-neighbours by an edge. Nodes carry typed metadata; edges carry
    cost information that depends on both endpoint risk levels.
    """

    def __init__(self, rows: int = 8, cols: int = 8):
        self.rows = rows
        self.cols = cols
        # NetworkX undirected graph - we attach our dataclasses as node/edge attrs.
        self.graph = nx.Graph()
        self.nodes: Dict[int, NodeData] = {}
        self.primary_hospital_id: Optional[int] = None
        self.ambulance_depot_id: Optional[int] = None
        self._build_grid()

    # ---------- construction helpers ----------

    def _build_grid(self) -> None:
        """Create one node per grid cell and connect 4-neighbours with edges."""
        for r in range(self.rows):
            for c in range(self.cols):
                node_id = self._cell_id(r, c)
                self.nodes[node_id] = NodeData(id=node_id, row=r, col=c)
                self.graph.add_node(node_id)

        # Connect 4-neighbours (right and down only - undirected handles the rest).
        for r in range(self.rows):
            for c in range(self.cols):
                u = self._cell_id(r, c)
                if c + 1 < self.cols:
                    self._add_edge(u, self._cell_id(r, c + 1))
                if r + 1 < self.rows:
                    self._add_edge(u, self._cell_id(r + 1, c))

    def _cell_id(self, row: int, col: int) -> int:
        return row * self.cols + col

    def _add_edge(self, u: int, v: int) -> None:
        edge = EdgeData()
        self.graph.add_edge(u, v, data=edge)

    # ---------- node accessors ----------

    def node(self, node_id: int) -> NodeData:
        return self.nodes[node_id]

    def cell(self, row: int, col: int) -> NodeData:
        return self.nodes[self._cell_id(row, col)]

    def all_nodes(self) -> List[NodeData]:
        return list(self.nodes.values())

    def nodes_of_type(self, loc_type: str) -> List[NodeData]:
        return [n for n in self.nodes.values() if n.type == loc_type]

    def position(self, node_id: int) -> Tuple[int, int]:
        n = self.nodes[node_id]
        return (n.row, n.col)

    # ---------- edge accessors ----------

    def edge(self, u: int, v: int) -> EdgeData:
        return self.graph[u][v]["data"]

    def all_edges(self) -> List[Tuple[int, int, EdgeData]]:
        return [(u, v, d["data"]) for u, v, d in self.graph.edges(data=True)]

    def neighbours(self, node_id: int) -> List[int]:
        return list(self.graph.neighbors(node_id))

    # ---------- mutation: types and costs ----------

    def set_node_type(self, node_id: int, loc_type: str) -> None:
        """Assign a location type and refresh costs of every adjacent edge."""
        self.nodes[node_id].type = loc_type
        # Residential roads have a lower base cost - update both sides.
        for nbr in self.neighbours(node_id):
            edge = self.edge(node_id, nbr)
            edge.base_cost = self._compute_base_cost(node_id, nbr)
            self._refresh_effective_cost(node_id, nbr, edge)

    def set_population_density(self, node_id: int, density: int) -> None:
        self.nodes[node_id].population_density = density

    def set_crime_risk(self, node_id: int, level: str) -> None:
        """Set crime risk level and recompute every incident edge's effective cost."""
        if level not in RISK_MULTIPLIER:
            raise ValueError(f"Unknown risk level: {level}")
        self.nodes[node_id].crime_risk_level = level
        for nbr in self.neighbours(node_id):
            edge = self.edge(node_id, nbr)
            self._refresh_effective_cost(node_id, nbr, edge)

    def block_edge(self, u: int, v: int, current_step: int = -1) -> None:
        edge = self.edge(u, v)
        edge.blocked = True
        edge.flash_until_step = current_step + 1  # flash for one step after blocking

    def unblock_edge(self, u: int, v: int) -> None:
        self.edge(u, v).blocked = False

    def reset_blocks(self) -> None:
        for _, _, edge in self.all_edges():
            edge.blocked = False

    # ---------- cost helpers ----------

    def _compute_base_cost(self, u: int, v: int) -> float:
        """Roads through residential zones cost 0.8; everything else costs 1.0."""
        a, b = self.nodes[u], self.nodes[v]
        if a.type == LOC_RESIDENTIAL or b.type == LOC_RESIDENTIAL:
            return COST_RESIDENTIAL
        return COST_STANDARD

    def _refresh_effective_cost(self, u: int, v: int, edge: EdgeData) -> None:
        """effective_cost = base_cost x max(risk multiplier of the two endpoints)."""
        a_risk = RISK_MULTIPLIER[self.nodes[u].crime_risk_level]
        b_risk = RISK_MULTIPLIER[self.nodes[v].crime_risk_level]
        edge.effective_cost = edge.base_cost * max(a_risk, b_risk)

    def refresh_all_costs(self) -> None:
        for u, v, edge in self.all_edges():
            edge.base_cost = self._compute_base_cost(u, v)
            self._refresh_effective_cost(u, v, edge)

    # ---------- traversal helpers shared by several modules ----------

    def bfs_hops(self, source: int, max_hops: Optional[int] = None) -> Dict[int, int]:
        """BFS hop counts from source. Treats blocked edges as impassable."""
        distances = {source: 0}
        queue: deque = deque([source])
        while queue:
            current = queue.popleft()
            if max_hops is not None and distances[current] >= max_hops:
                continue
            for nbr in self.neighbours(current):
                if self.edge(current, nbr).blocked:
                    continue
                if nbr not in distances:
                    distances[nbr] = distances[current] + 1
                    queue.append(nbr)
        return distances

    def hops_ignoring_blocks(self, source: int, max_hops: Optional[int] = None) -> Dict[int, int]:
        """BFS that ignores blocked edges - used during initial layout planning."""
        distances = {source: 0}
        queue: deque = deque([source])
        while queue:
            current = queue.popleft()
            if max_hops is not None and distances[current] >= max_hops:
                continue
            for nbr in self.neighbours(current):
                if nbr not in distances:
                    distances[nbr] = distances[current] + 1
                    queue.append(nbr)
        return distances

    # ---------- snapshot / debug ----------

    def type_grid(self) -> List[List[str]]:
        """Return a rows x cols grid of letter codes - convenient for the UI."""
        grid = [[LETTER_CODE[LOC_EMPTY]] * self.cols for _ in range(self.rows)]
        for n in self.nodes.values():
            grid[n.row][n.col] = LETTER_CODE.get(n.type, "?")
        return grid

    def node_count_of_type(self, loc_type: str) -> int:
        return sum(1 for n in self.nodes.values() if n.type == loc_type)
