from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .city_graph import (
    CityGraph,
    LOCATION_TYPES,
    LOC_AMBULANCE_DEPOT,
    LOC_EMPTY,
    LOC_HOSPITAL,
    LOC_INDUSTRIAL,
    LOC_POWER_PLANT,
    LOC_RESIDENTIAL,
    LOC_SCHOOL,
)


DEFAULT_QUOTAS = {
    LOC_HOSPITAL: 2,
    LOC_AMBULANCE_DEPOT: 1,
    LOC_INDUSTRIAL: 3,
    LOC_POWER_PLANT: 1,
    LOC_SCHOOL: 2,
}

POPULATION_DENSITY_RANGE = (50, 300)  # citizens per residential cell

RESIDENTIAL_HOSPITAL_HOPS = 3
POWERPLANT_INDUSTRIAL_HOPS = 2


@dataclass
class LayoutResult:
    """Outcome of a layout-planning run."""

    success: bool
    assignment: Dict[int, str]
    violated_rule: Optional[str] = None  # populated when min-conflicts is used
    iterations: int = 0
    used_min_conflicts: bool = False


class CityLayoutPlanner:
    """CSP planner that fills the city graph with location types."""

    def __init__(self, graph: CityGraph, quotas: Optional[Dict[str, int]] = None,
                 seed: Optional[int] = None):
        self.graph = graph
        self.quotas = dict(quotas or DEFAULT_QUOTAS)
        self.rng = random.Random(seed)
        self.domains: Dict[int, Set[str]] = {
            n.id: set(LOCATION_TYPES) | {LOC_EMPTY} for n in graph.all_nodes()
        }
        self._total_backtracks = 0


    def solve(self) -> LayoutResult:
        """Run CSP backtracking; fall back to Min-Conflicts if it fails."""
        self._seed_quota_assignments()
        self._ac3()

        assignment: Dict[int, str] = {
            nid: next(iter(d)) for nid, d in self.domains.items() if len(d) == 1
        }
        result = self._backtrack(assignment, iterations=0)

        if result is None:
            mc_result = self._min_conflicts()
            self._apply(mc_result.assignment)
            self._designate_primary_hospital()
            return mc_result

        self._apply(result)
        self._designate_primary_hospital()
        return LayoutResult(success=True, assignment=result, iterations=0)


    def _seed_quota_assignments(self) -> None:
        """Pre-assign required location types to random empty cells.

        Without this step the CSP could happily declare every cell Residential
        and call itself done. Seeding ensures hospitals, industrial zones, etc.
        actually appear before backtracking starts.
        """
        cells = [n.id for n in self.graph.all_nodes()]
        self.rng.shuffle(cells)
        idx = 0
        for loc_type, count in self.quotas.items():
            for _ in range(count):
                if idx >= len(cells):
                    break
                cell = cells[idx]
                idx += 1
                self.domains[cell] = {loc_type}
        for cell in cells[idx:]:
            self.domains[cell] = {LOC_RESIDENTIAL, LOC_EMPTY}


    def _ac3(self) -> None:
        """Prune values that can never satisfy the industrial-adjacency rule."""
        queue: deque = deque(
            (u, v) for u in self.domains for v in self.graph.neighbours(u)
        )
        while queue:
            xi, xj = queue.popleft()
            if self._revise(xi, xj):
                if not self.domains[xi]:
                    return  # caller will detect failure
                for xk in self.graph.neighbours(xi):
                    if xk != xj:
                        queue.append((xk, xi))

    def _revise(self, xi: int, xj: int) -> bool:
        """Remove any value in xi that has no consistent partner in xj."""
        revised = False
        for value in list(self.domains[xi]):
            if not any(self._adjacency_ok(value, other) for other in self.domains[xj]):
                self.domains[xi].remove(value)
                revised = True
        return revised

    @staticmethod
    def _adjacency_ok(a: str, b: str) -> bool:
        """Industrial cells must not sit next to schools or hospitals."""
        bad_neighbours = {LOC_SCHOOL, LOC_HOSPITAL}
        if a == LOC_INDUSTRIAL and b in bad_neighbours:
            return False
        if b == LOC_INDUSTRIAL and a in bad_neighbours:
            return False
        return True


    def _backtrack(self, assignment: Dict[int, str], iterations: int,
                   max_iterations: int = 5_000) -> Optional[Dict[int, str]]:
        self._total_backtracks += 1
        if self._total_backtracks > max_iterations:
            return None
        if len(assignment) == len(self.domains):
            if self._global_constraints_ok(assignment):
                return assignment
            return None

        unassigned = [nid for nid in self.domains if nid not in assignment]
        var = min(unassigned, key=lambda v: len(self.domains[v]))

        for value in list(self.domains[var]):
            if not self._consistent_with_neighbours(var, value, assignment):
                continue
            saved_domains = {nid: set(d) for nid, d in self.domains.items()}
            assignment[var] = value
            if self._forward_check(var, value):
                result = self._backtrack(assignment, iterations + 1, max_iterations)
                if result is not None:
                    return result
            assignment.pop(var)
            self.domains = saved_domains
        return None

    def _consistent_with_neighbours(self, var: int, value: str,
                                    assignment: Dict[int, str]) -> bool:
        for nbr in self.graph.neighbours(var):
            if nbr in assignment and not self._adjacency_ok(value, assignment[nbr]):
                return False
        return True

    def _forward_check(self, var: int, value: str) -> bool:
        """Remove neighbour values that would violate adjacency given this pick."""
        for nbr in self.graph.neighbours(var):
            for cand in list(self.domains[nbr]):
                if not self._adjacency_ok(value, cand):
                    self.domains[nbr].discard(cand)
            if not self.domains[nbr]:
                return False
        return True


    def _global_constraints_ok(self, assignment: Dict[int, str]) -> bool:
        return (
            self._residential_within_hospital_hops(assignment)
            and self._powerplants_within_industrial_hops(assignment)
        )

    def _residential_within_hospital_hops(self, assignment: Dict[int, str]) -> bool:
        hospitals = [nid for nid, t in assignment.items() if t == LOC_HOSPITAL]
        if not hospitals:
            return not any(t == LOC_RESIDENTIAL for t in assignment.values())
        covered = set()
        for h in hospitals:
            covered |= set(self._bfs_hops_via_assignment(h, assignment,
                                                        RESIDENTIAL_HOSPITAL_HOPS).keys())
        for nid, t in assignment.items():
            if t == LOC_RESIDENTIAL and nid not in covered:
                return False
        return True

    def _powerplants_within_industrial_hops(self, assignment: Dict[int, str]) -> bool:
        industrials = [nid for nid, t in assignment.items() if t == LOC_INDUSTRIAL]
        if not industrials:
            return not any(t == LOC_POWER_PLANT for t in assignment.values())
        covered = set()
        for i in industrials:
            covered |= set(self._bfs_hops_via_assignment(i, assignment,
                                                        POWERPLANT_INDUSTRIAL_HOPS).keys())
        for nid, t in assignment.items():
            if t == LOC_POWER_PLANT and nid not in covered:
                return False
        return True

    def _bfs_hops_via_assignment(self, source: int, assignment: Dict[int, str],
                                 max_hops: int) -> Dict[int, int]:
        """BFS from source, capped at max_hops. Used during search before the graph is mutated."""
        distances = {source: 0}
        queue: deque = deque([source])
        while queue:
            curr = queue.popleft()
            if distances[curr] >= max_hops:
                continue
            for nbr in self.graph.neighbours(curr):
                if nbr not in distances:
                    distances[nbr] = distances[curr] + 1
                    queue.append(nbr)
        return distances


    def _min_conflicts(self, max_steps: int = 1000) -> LayoutResult:
        """When backtracking fails, find the least-violated complete assignment."""
        assignment = self._random_seed_assignment()
        best = dict(assignment)
        best_score, _ = self._violation_score(best)

        for step in range(max_steps):
            score, per_rule = self._violation_score(assignment)
            if score == 0:
                return LayoutResult(success=True, assignment=assignment,
                                    iterations=step, used_min_conflicts=True)
            if score < best_score:
                best, best_score = dict(assignment), score

            conflicting = self._conflicting_cells(assignment)
            if not conflicting:
                break
            cell = self.rng.choice(conflicting)
            best_value = self._least_conflict_value(cell, assignment)
            assignment[cell] = best_value

        worst_rule = max(self._violation_score(best)[1].items(),
                         key=lambda kv: kv[1], default=("none", 0))[0]
        return LayoutResult(success=False, assignment=best,
                            violated_rule=worst_rule, iterations=max_steps,
                            used_min_conflicts=True)

    def _random_seed_assignment(self) -> Dict[int, str]:
        """Fill the grid with random types respecting the configured quotas."""
        cells = [n.id for n in self.graph.all_nodes()]
        self.rng.shuffle(cells)
        assignment = {c: LOC_RESIDENTIAL for c in cells}
        idx = 0
        for loc_type, count in self.quotas.items():
            for _ in range(count):
                if idx >= len(cells):
                    break
                assignment[cells[idx]] = loc_type
                idx += 1
        return assignment

    def _violation_score(self, assignment: Dict[int, str]) -> Tuple[int, Dict[str, int]]:
        """Total violations and a per-rule breakdown."""
        rule_counts = {"industrial_adjacency": 0,
                       "residential_hospital_hops": 0,
                       "powerplant_industrial_hops": 0}

        for u, v, _ in self.graph.all_edges():
            if not self._adjacency_ok(assignment[u], assignment[v]):
                rule_counts["industrial_adjacency"] += 1

        hospitals = [nid for nid, t in assignment.items() if t == LOC_HOSPITAL]
        residential_cells = [nid for nid, t in assignment.items() if t == LOC_RESIDENTIAL]
        if hospitals:
            covered = set()
            for h in hospitals:
                covered |= self._bfs_hops_via_assignment(
                    h, assignment, RESIDENTIAL_HOSPITAL_HOPS).keys()
            rule_counts["residential_hospital_hops"] = sum(
                1 for r in residential_cells if r not in covered)
        else:
            rule_counts["residential_hospital_hops"] = len(residential_cells)

        industrials = [nid for nid, t in assignment.items() if t == LOC_INDUSTRIAL]
        plants = [nid for nid, t in assignment.items() if t == LOC_POWER_PLANT]
        if industrials:
            covered = set()
            for i in industrials:
                covered |= self._bfs_hops_via_assignment(
                    i, assignment, POWERPLANT_INDUSTRIAL_HOPS).keys()
            rule_counts["powerplant_industrial_hops"] = sum(
                1 for p in plants if p not in covered)
        else:
            rule_counts["powerplant_industrial_hops"] = len(plants)

        return sum(rule_counts.values()), rule_counts

    def _conflicting_cells(self, assignment: Dict[int, str]) -> List[int]:
        bad: Set[int] = set()
        for u, v, _ in self.graph.all_edges():
            if not self._adjacency_ok(assignment[u], assignment[v]):
                bad.add(u)
                bad.add(v)

        hospitals = [nid for nid, t in assignment.items() if t == LOC_HOSPITAL]
        if hospitals:
            covered = set()
            for h in hospitals:
                covered |= self._bfs_hops_via_assignment(
                    h, assignment, RESIDENTIAL_HOSPITAL_HOPS).keys()
            for nid, t in assignment.items():
                if t == LOC_RESIDENTIAL and nid not in covered:
                    bad.add(nid)

        industrials = [nid for nid, t in assignment.items() if t == LOC_INDUSTRIAL]
        if industrials:
            covered = set()
            for i in industrials:
                covered |= self._bfs_hops_via_assignment(
                    i, assignment, POWERPLANT_INDUSTRIAL_HOPS).keys()
            for nid, t in assignment.items():
                if t == LOC_POWER_PLANT and nid not in covered:
                    bad.add(nid)

        return list(bad)

    def _least_conflict_value(self, cell: int, assignment: Dict[int, str]) -> str:
        """Try every type for this cell and pick whichever drops total violations most."""
        original = assignment[cell]
        best_value = original
        best_score, _ = self._violation_score(assignment)
        for value in LOCATION_TYPES:
            assignment[cell] = value
            score, _ = self._violation_score(assignment)
            if score < best_score:
                best_score = score
                best_value = value
        assignment[cell] = original
        return best_value


    def _apply(self, assignment: Dict[int, str]) -> None:
        for nid, loc_type in assignment.items():
            self.graph.set_node_type(nid, loc_type)
            if loc_type == LOC_RESIDENTIAL:
                self.graph.set_population_density(
                    nid, self.rng.randint(*POPULATION_DENSITY_RANGE))
            elif loc_type in (LOC_HOSPITAL, LOC_SCHOOL, LOC_INDUSTRIAL):
                self.graph.set_population_density(nid, self.rng.randint(20, 80))
            else:
                self.graph.set_population_density(nid, 0)
        depots = self.graph.nodes_of_type(LOC_AMBULANCE_DEPOT)
        if depots:
            self.graph.ambulance_depot_id = depots[0].id

    def _designate_primary_hospital(self) -> None:
        """Pick the hospital that 'covers' the most residential population."""
        hospitals = self.graph.nodes_of_type(LOC_HOSPITAL)
        if not hospitals:
            self.graph.primary_hospital_id = None
            return

        best_hospital = None
        best_coverage = -1
        for h in hospitals:
            distances = self.graph.hops_ignoring_blocks(h.id, max_hops=RESIDENTIAL_HOSPITAL_HOPS)
            coverage = sum(self.graph.node(nid).population_density
                           for nid in distances
                           if self.graph.node(nid).type == LOC_RESIDENTIAL)
            if coverage > best_coverage:
                best_coverage = coverage
                best_hospital = h

        if best_hospital is not None:
            best_hospital.is_primary_hospital = True
            self.graph.primary_hospital_id = best_hospital.id


def plan_layout(graph: CityGraph, quotas: Optional[Dict[str, int]] = None,
                seed: Optional[int] = None) -> LayoutResult:
    """Convenience entry point for the simulation loop."""
    return CityLayoutPlanner(graph, quotas=quotas, seed=seed).solve()
