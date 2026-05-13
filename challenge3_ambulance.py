from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .city_graph import CityGraph, LOC_EMPTY, LOC_RESIDENTIAL


# GA hyperparameters - exposed at module top so they can be tuned without
# touching algorithm logic.
POPULATION_SIZE = 30
GENERATIONS = 40
MUTATION_RATE = 0.25
TOURNAMENT_K = 3
ELITISM = 2
NUM_AMBULANCES = 3


@dataclass
class AmbulanceResult:
    """Outcome of a placement run."""

    positions: List[int]
    worst_case_distance: float
    generations_run: int


class AmbulancePlanner:
    """GA-based ambulance placement planner."""

    def __init__(self, graph: CityGraph, seed: Optional[int] = None):
        self.graph = graph
        self.rng = random.Random(seed)
        self._candidate_pool = self._build_candidate_pool()
        self._residentials = [n.id for n in graph.all_nodes() if n.type == LOC_RESIDENTIAL]
        self._cached_view: Optional[nx.Graph] = None

    def _build_candidate_pool(self) -> List[int]:
        """Any non-empty cell is a valid ambulance posting site."""
        return [n.id for n in self.graph.all_nodes() if n.type != LOC_EMPTY]

    # ------------------------------------------------------------------
    # GA driver
    # ------------------------------------------------------------------

    def solve(self) -> AmbulanceResult:
        if len(self._candidate_pool) < NUM_AMBULANCES or not self._residentials:
            # Degenerate case: no residentials or not enough cells. Fall back
            # gracefully by placing ambulances at the first available cells.
            fallback = self._candidate_pool[:NUM_AMBULANCES]
            return AmbulanceResult(positions=fallback, worst_case_distance=math.inf,
                                   generations_run=0)

        population = [self._random_chromosome() for _ in range(POPULATION_SIZE)]
        best = min(population, key=self._fitness)
        best_fitness = self._fitness(best)

        for gen in range(GENERATIONS):
            new_population = sorted(population, key=self._fitness)[:ELITISM]
            while len(new_population) < POPULATION_SIZE:
                parent_a = self._tournament(population)
                parent_b = self._tournament(population)
                child = self._crossover(parent_a, parent_b)
                if self.rng.random() < MUTATION_RATE:
                    child = self._mutate(child)
                new_population.append(child)
            population = new_population

            current_best = min(population, key=self._fitness)
            current_fitness = self._fitness(current_best)
            if current_fitness < best_fitness:
                best, best_fitness = current_best, current_fitness

        return AmbulanceResult(positions=list(best),
                               worst_case_distance=best_fitness,
                               generations_run=GENERATIONS)

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------

    def _random_chromosome(self) -> Tuple[int, ...]:
        return tuple(self.rng.sample(self._candidate_pool, NUM_AMBULANCES))

    def _tournament(self, population: List[Tuple[int, ...]]) -> Tuple[int, ...]:
        contestants = self.rng.sample(population, TOURNAMENT_K)
        return min(contestants, key=self._fitness)

    def _crossover(self, parent_a: Tuple[int, ...],
                   parent_b: Tuple[int, ...]) -> Tuple[int, ...]:
        """Single-point crossover with duplicate repair."""
        cut = self.rng.randint(1, NUM_AMBULANCES - 1)
        child = list(parent_a[:cut]) + [g for g in parent_b if g not in parent_a[:cut]]
        # Pad with random unique candidates if we didn't get enough genes.
        while len(child) < NUM_AMBULANCES:
            cand = self.rng.choice(self._candidate_pool)
            if cand not in child:
                child.append(cand)
        return tuple(child[:NUM_AMBULANCES])

    def _mutate(self, chromosome: Tuple[int, ...]) -> Tuple[int, ...]:
        """Replace one gene with a random unused candidate."""
        gene_index = self.rng.randint(0, NUM_AMBULANCES - 1)
        replacement = self.rng.choice(self._candidate_pool)
        attempts = 0
        # Avoid duplicates; bail out after a few tries if everything collides.
        while replacement in chromosome and attempts < 10:
            replacement = self.rng.choice(self._candidate_pool)
            attempts += 1
        new = list(chromosome)
        new[gene_index] = replacement
        return tuple(new)

    # ------------------------------------------------------------------
    # Fitness = worst-case Dijkstra distance from any residential to nearest amb.
    # ------------------------------------------------------------------

    def _fitness(self, chromosome: Tuple[int, ...]) -> float:
        if self._cached_view is None:
            self._cached_view = self._weighted_view()
        weighted = self._cached_view
        ambulance_distances: Dict[int, Dict[int, float]] = {}
        for amb in chromosome:
            try:
                ambulance_distances[amb] = nx.single_source_dijkstra_path_length(
                    weighted, amb)
            except nx.NodeNotFound:
                ambulance_distances[amb] = {}

        worst = 0.0
        for r in self._residentials:
            best_for_this_resident = math.inf
            for amb_dist in ambulance_distances.values():
                if r in amb_dist and amb_dist[r] < best_for_this_resident:
                    best_for_this_resident = amb_dist[r]
            if best_for_this_resident == math.inf:
                # Unreachable - heavy penalty so the GA strongly avoids it.
                return 1e6
            if best_for_this_resident > worst:
                worst = best_for_this_resident
        return worst

    def _weighted_view(self) -> nx.Graph:
        """Snapshot the graph with effective_cost weights and blocked edges removed.

        We rebuild this each fitness call because the GA may run after road
        blocks or risk updates. Caching is handled at a higher level (the
        simulation triggers a re-run when the graph actually changes).
        """
        view = nx.Graph()
        for n in self.graph.all_nodes():
            view.add_node(n.id)
        for u, v, edge in self.graph.all_edges():
            if edge.blocked:
                continue
            view.add_edge(u, v, weight=edge.effective_cost)
        return view


def place_ambulances(graph: CityGraph, seed: Optional[int] = None) -> AmbulanceResult:
    """Convenience entry point for the simulation loop."""
    return AmbulancePlanner(graph, seed=seed).solve()
