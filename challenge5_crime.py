from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .city_graph import (
    CityGraph,
    LOC_EMPTY,
    LOC_INDUSTRIAL,
    LOC_RESIDENTIAL,
    RISK_MULTIPLIER,
)


NUM_CLUSTERS = 3
NUM_TREES = 60
NUM_POLICE = 10
SYNTHETIC_NOISE = 0.05  # adds light label noise so the RF actually has to learn

HIGH_DENSITY_THRESHOLD = 0.66  # quantile cutoff after standardisation
LOW_DENSITY_THRESHOLD = 0.33


@dataclass
class CrimeResult:
    """Outcome of running the full ML pipeline."""

    cluster_labels: Dict[int, int]
    risk_levels: Dict[int, str]
    police_deployment: List[int] = field(default_factory=list)
    accuracy: float = 0.0           # train accuracy, sanity-check only
    cluster_centers: Optional[np.ndarray] = None


class CrimeRiskPipeline:
    """K-Means clustering followed by Random Forest classification."""

    def __init__(self, graph: CityGraph, seed: Optional[int] = None):
        self.graph = graph
        self.seed = seed
        self._industrial_distance_cache: Dict[int, int] = {}


    def run(self) -> CrimeResult:
        try:
            return self._run_internal()
        except Exception as exc:
            print(f"[challenge5] pipeline failed: {exc}")
            return CrimeResult(cluster_labels={}, risk_levels={})

    def _run_internal(self) -> CrimeResult:
        node_ids, features = self._build_feature_matrix()
        if len(node_ids) < NUM_CLUSTERS:
            return CrimeResult(cluster_labels={}, risk_levels={})

        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=NUM_CLUSTERS, n_init=10, random_state=self.seed)
        cluster_ids = kmeans.fit_predict(scaled)
        cluster_labels = {nid: int(c) for nid, c in zip(node_ids, cluster_ids)}

        synthetic_labels = self._synthesize_labels(scaled, cluster_ids)
        rf = RandomForestClassifier(n_estimators=NUM_TREES, random_state=self.seed)
        rf.fit(scaled, synthetic_labels)
        predictions = rf.predict(scaled)
        accuracy = float(np.mean(predictions == synthetic_labels))

        risk_levels = {nid: str(level) for nid, level in zip(node_ids, predictions)}

        for nid, level in risk_levels.items():
            self.graph.set_crime_risk(nid, level)
        for n in self.graph.all_nodes():
            if n.id not in risk_levels:
                self.graph.set_crime_risk(n.id, "Low")

        police = self._deploy_police(rf, scaled, node_ids)

        return CrimeResult(
            cluster_labels=cluster_labels,
            risk_levels=risk_levels,
            police_deployment=police,
            accuracy=accuracy,
            cluster_centers=kmeans.cluster_centers_,
        )


    def _build_feature_matrix(self) -> Tuple[List[int], np.ndarray]:
        """Return (node_ids, [[density, industrial_proximity], ...]).

        We include every non-empty cell so the model assigns a risk level to
        infrastructure too - hospitals in high-crime areas are still risky.
        """
        node_ids: List[int] = []
        rows: List[List[float]] = []
        for n in self.graph.all_nodes():
            if n.type == LOC_EMPTY:
                continue
            node_ids.append(n.id)
            density = float(n.population_density)
            proximity = float(self._industrial_proximity(n.id))
            rows.append([density, proximity])
        return node_ids, np.array(rows, dtype=float)

    def _industrial_proximity(self, node_id: int) -> int:
        """Hops to the nearest industrial cell (capped). Cached per call."""
        if node_id in self._industrial_distance_cache:
            return self._industrial_distance_cache[node_id]

        industrials = [n.id for n in self.graph.all_nodes() if n.type == LOC_INDUSTRIAL]
        if not industrials:
            self._industrial_distance_cache[node_id] = 99
            return 99

        if not self._industrial_distance_cache:
            distances: Dict[int, int] = {i: 0 for i in industrials}
            queue: deque = deque(industrials)
            while queue:
                curr = queue.popleft()
                for nbr in self.graph.neighbours(curr):
                    if nbr not in distances:
                        distances[nbr] = distances[curr] + 1
                        queue.append(nbr)
            for nid, d in distances.items():
                self._industrial_distance_cache[nid] = d
        return self._industrial_distance_cache.get(node_id, 99)


    def _synthesize_labels(self, scaled: np.ndarray,
                           cluster_ids: np.ndarray) -> np.ndarray:
        """Generate High/Medium/Low labels using urban-criminology heuristics.

        scaled[:, 0] = standardised density, scaled[:, 1] = standardised
        proximity (low value = closer to industry). Dense + close = High;
        sparse + far = Low; everything else = Medium. A small amount of label
        noise is added so the RF actually has a learning signal.
        """
        rng = np.random.default_rng(self.seed)
        density = scaled[:, 0]
        proximity = scaled[:, 1]
        risk_score = density - proximity  # subtract because lower proximity = closer
        high_cut = np.quantile(risk_score, HIGH_DENSITY_THRESHOLD)
        low_cut = np.quantile(risk_score, LOW_DENSITY_THRESHOLD)

        labels = np.empty(len(risk_score), dtype=object)
        for i, score in enumerate(risk_score):
            if score >= high_cut:
                labels[i] = "High"
            elif score <= low_cut:
                labels[i] = "Low"
            else:
                labels[i] = "Medium"

        for i in range(len(labels)):
            if rng.random() < SYNTHETIC_NOISE:
                labels[i] = rng.choice(list(RISK_MULTIPLIER.keys()))

        return labels


    def _deploy_police(self, rf: RandomForestClassifier, scaled: np.ndarray,
                       node_ids: List[int]) -> List[int]:
        """Place 10 officers at the highest-risk locations."""
        if not node_ids:
            return []
        proba = rf.predict_proba(scaled)
        try:
            high_idx = list(rf.classes_).index("High")
            scores = proba[:, high_idx]
        except ValueError:
            scores = proba[:, 0]

        for i, nid in enumerate(node_ids):
            if self.graph.node(nid).type == LOC_RESIDENTIAL:
                scores[i] += 0.05

        ranked = sorted(zip(node_ids, scores), key=lambda kv: kv[1], reverse=True)
        return [nid for nid, _ in ranked[:NUM_POLICE]]


def predict_crime_risk(graph: CityGraph, seed: Optional[int] = None) -> CrimeResult:
    """Convenience entry point for the simulation loop."""
    return CrimeRiskPipeline(graph, seed=seed).run()
