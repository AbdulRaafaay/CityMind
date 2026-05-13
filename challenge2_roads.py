from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from .city_graph import CityGraph, LOC_EMPTY


@dataclass
class RoadNetworkResult:
    """Outcome of road planning."""

    selected_edges: Set[Tuple[int, int]]
    total_cost: float
    edge_disjoint_paths: int  # final flow between hospital and depot
    extra_edges_added: int    # how many edges beyond MST we added for redundancy


class RoadNetworkBuilder:
    """Build the road network on the shared city graph."""

    def __init__(self, graph: CityGraph):
        self.graph = graph

    def build(self) -> RoadNetworkResult:
        candidates = sorted(
            ((u, v, edge.base_cost) for u, v, edge in self.graph.all_edges()
             if not (self.graph.node(u).type == LOC_EMPTY
                     and self.graph.node(v).type == LOC_EMPTY)),
            key=lambda triple: triple[2],
        )

        parent: Dict[int, int] = {}
        rank: Dict[int, int] = {}
        selected: Set[Tuple[int, int]] = set()
        total_cost = 0.0

        relevant_nodes = {n.id for n in self.graph.all_nodes() if n.type != LOC_EMPTY}
        for nid in relevant_nodes:
            parent[nid] = nid
            rank[nid] = 0

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(a: int, b: int) -> bool:
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True

        target_tree_size = len(relevant_nodes) - 1
        leftover_candidates: List[Tuple[int, int, float]] = []

        for u, v, cost in candidates:
            if u not in relevant_nodes or v not in relevant_nodes:
                leftover_candidates.append((u, v, cost))
                continue
            if union(u, v):
                selected.add(self._normalise(u, v))
                total_cost += cost
                if len(selected) == target_tree_size:
                    break
            else:
                leftover_candidates.append((u, v, cost))

        already_seen = {self._normalise(u, v) for u, v, _ in leftover_candidates}
        for u, v, cost in candidates:
            key = self._normalise(u, v)
            if key not in selected and key not in already_seen:
                leftover_candidates.append((u, v, cost))
                already_seen.add(key)
        leftover_candidates.sort(key=lambda triple: triple[2])


        primary_h = self.graph.primary_hospital_id
        depot = self.graph.ambulance_depot_id
        extra_edges: List[Tuple[int, int, float]] = []
        extra_added = 0
        flow = self._edge_disjoint_count(primary_h, depot, selected)

        if primary_h is not None and depot is not None and flow < 2:
            for u, v, cost in leftover_candidates:
                key = self._normalise(u, v)
                if key in selected:
                    continue
                if (self.graph.node(u).type == LOC_EMPTY
                        and self.graph.node(v).type == LOC_EMPTY):
                    continue
                selected.add(key)
                extra_edges.append((u, v, cost))
                flow = self._edge_disjoint_count(primary_h, depot, selected)
                if flow >= 2:
                    break

            for u, v, cost in sorted(extra_edges, key=lambda t: -t[2]):
                key = self._normalise(u, v)
                if key not in selected:
                    continue
                selected.discard(key)
                if self._edge_disjoint_count(primary_h, depot, selected) < 2:
                    selected.add(key)  # rollback - this edge was load-bearing

            extra_added = sum(1 for u, v, _ in extra_edges
                              if self._normalise(u, v) in selected)
            total_cost += sum(cost for u, v, cost in extra_edges
                              if self._normalise(u, v) in selected)

        for u, v, edge in self.graph.all_edges():
            edge.blocked = self._normalise(u, v) not in selected
        flow = self._edge_disjoint_count(primary_h, depot, selected)

        return RoadNetworkResult(
            selected_edges=selected,
            total_cost=total_cost,
            edge_disjoint_paths=flow,
            extra_edges_added=extra_added,
        )


    @staticmethod
    def _normalise(u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u <= v else (v, u)

    def _edge_disjoint_count(self, source: Optional[int], sink: Optional[int],
                             selected: Set[Tuple[int, int]]) -> int:
        """Maximum number of edge-disjoint paths between source and sink.

        We build the undirected subgraph of selected edges and ask NetworkX
        for the local edge connectivity. By Menger's theorem this equals the
        number of edge-disjoint source-sink paths.
        """
        if source is None or sink is None or source == sink:
            return 0
        graph = nx.Graph()
        for u, v in selected:
            graph.add_edge(u, v)
        if source not in graph or sink not in graph:
            return 0
        return int(nx.edge_connectivity(graph, source, sink))


def build_road_network(graph: CityGraph) -> RoadNetworkResult:
    """Convenience entry point for the simulation loop."""
    return RoadNetworkBuilder(graph).build()
