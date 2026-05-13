from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .city_graph import CityGraph, COST_RESIDENTIAL


HEURISTIC_SCALE = COST_RESIDENTIAL  # admissible because no edge is cheaper


@dataclass
class RouteSegment:
    """One leg of the journey: source -> destination via this path."""

    source: int
    target: int
    path: List[int]
    cost: float
    replanned: bool = False  # set True when the segment was recomputed mid-journey


@dataclass
class RoutingResult:
    """Outcome of a multi-civilian routing run."""

    segments: List[RouteSegment] = field(default_factory=list)
    visited: List[int] = field(default_factory=list)
    unreachable: List[int] = field(default_factory=list)
    total_cost: float = 0.0


class EmergencyRouter:
    """A* router with Nearest-Neighbour visit ordering and live re-planning."""

    def __init__(self, graph: CityGraph):
        self.graph = graph


    def a_star(self, source: int, target: int) -> Tuple[List[int], float]:
        """Standard A* using effective_cost. Returns (path, cost) or ([], inf)."""
        if source == target:
            return [source], 0.0

        open_heap: List[Tuple[float, int]] = []
        heapq.heappush(open_heap, (0.0, source))
        came_from: Dict[int, int] = {}
        g_score: Dict[int, float] = {source: 0.0}

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == target:
                return self._reconstruct(came_from, current), g_score[current]

            for nbr in self.graph.neighbours(current):
                edge = self.graph.edge(current, nbr)
                if edge.blocked:
                    continue
                tentative_g = g_score[current] + edge.effective_cost
                if tentative_g < g_score.get(nbr, math.inf):
                    came_from[nbr] = current
                    g_score[nbr] = tentative_g
                    f_score = tentative_g + self._heuristic(nbr, target)
                    heapq.heappush(open_heap, (f_score, nbr))

        return [], math.inf

    def _heuristic(self, a: int, b: int) -> float:
        ra, ca = self.graph.position(a)
        rb, cb = self.graph.position(b)
        return (abs(ra - rb) + abs(ca - cb)) * HEURISTIC_SCALE

    @staticmethod
    def _reconstruct(came_from: Dict[int, int], current: int) -> List[int]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


    def route_through(self, start: int, civilians: List[int],
                      block_callback=None) -> RoutingResult:
        """Visit every civilian starting from `start`.

        block_callback(current_node) is invoked just before each step is
        traversed and may block edges on the shared graph; this simulates a
        road flooding while the team is en route. We detect it by re-running
        A* whenever the current segment's next edge has become blocked.
        """
        result = RoutingResult()
        unvisited = list(civilians)
        position = start

        while unvisited:
            target = self._nearest_unvisited(position, unvisited)
            if target is None:
                result.unreachable.extend(unvisited)
                break

            path, cost = self.a_star(position, target)
            if not path or cost == math.inf:
                result.unreachable.append(target)
                unvisited.remove(target)
                continue

            segment = self._walk_segment(position, target, path, cost, block_callback)
            result.segments.append(segment)
            position = target
            result.visited.append(target)
            unvisited.remove(target)
            result.total_cost += segment.cost

        return result

    def _walk_segment(self, source: int, target: int, path: List[int],
                      cost: float, block_callback) -> RouteSegment:
        """Walk a planned path one edge at a time, replanning if edges block."""
        full_path = list(path)
        total_cost = 0.0
        replanned = False
        idx = 0
        while idx < len(full_path) - 1:
            here = full_path[idx]
            nxt = full_path[idx + 1]

            if block_callback is not None:
                block_callback(here)

            edge = self.graph.edge(here, nxt)
            if edge.blocked:
                new_path, new_cost = self.a_star(here, target)
                if not new_path or new_cost == math.inf:
                    return RouteSegment(source=source, target=target,
                                        path=full_path[:idx + 1],
                                        cost=total_cost + math.inf,
                                        replanned=True)
                replanned = True
                full_path = full_path[:idx + 1] + new_path[1:]
                continue

            total_cost += edge.effective_cost
            idx += 1

        return RouteSegment(source=source, target=target, path=full_path,
                            cost=total_cost, replanned=replanned)

    def _nearest_unvisited(self, position: int,
                           unvisited: List[int]) -> Optional[int]:
        """Pick the unvisited civilian with the shortest A* distance from here."""
        best_target = None
        best_cost = math.inf
        for civilian in unvisited:
            _, cost = self.a_star(position, civilian)
            if cost < best_cost:
                best_cost = cost
                best_target = civilian
        return best_target


def shortest_path(graph: CityGraph, source: int,
                  target: int) -> Tuple[List[int], float]:
    """Helper used by other modules that want a quick A* lookup."""
    return EmergencyRouter(graph).a_star(source, target)
