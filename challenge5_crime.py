"""Challenge 5: Crime Risk Prediction and Integration.

Two-stage ML pipeline:
 1. K-Means (unsupervised) clusters neighbourhoods by [population_density,
    industrial_proximity]. Features are standardised first so density does
    not dominate proximity numerically.
 2. Random Forest (supervised) is trained on synthetic labels: dense and
    industrial-adjacent areas are tagged High, low/quiet areas Low,
    everything else Medium. The trained model then assigns High/Medium/Low
    to every node in the graph.

The predicted risk levels are written back into the shared graph, which
recomputes effective_cost on every incident edge using the multipliers from
the design doc (High 1.5x, Medium 1.2x, Low 1.0x, max of two endpoints).

Police deployment: the city has 10 officers. We place them at the 10 nodes
with the highest predicted risk score, breaking ties by population density.
"""

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


# Pipeline configuration.
NUM_CLUSTERS = 3
NUM_TREES = 60
NUM_POLICE = 10
SYNTHETIC_NOISE = 0.05  # adds light label noise so the RF actually has to learn

# Feature thresholds for synthetic label generation. These reflect the urban
# criminology heuristic in the design doc: dense + close to industry => high.
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

    # ------------------------------------------------------------------
    # Pipeline driver
    # ------------------------------------------------------------------

    def run(self) -> CrimeResult:
        try:
            return self._run_internal()
        except Exception as exc:
            # ML libraries occasionally raise on degenerate inputs; we don't
            # want a single bad run to take down the simulation.
            print(f"[challenge5] pipeline failed: {exc}")
            return CrimeResult(cluster_labels={}, risk_levels={})

    def _run_internal(self) -> CrimeResult:
        node_ids, features = self._build_feature_matrix()
        if len(node_ids) < NUM_CLUSTERS:
            return CrimeResult(cluster_labels={}, risk_levels={})

        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        # Step 1 - unsupervised clustering.
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, n_init=10, random_state=self.seed)
        cluster_ids = kmeans.fit_predict(scaled)
        cluster_labels = {nid: int(c) for nid, c in zip(node_ids, cluster_ids)}

        # Step 2 - generate synthetic labels and train the supervised model.
        synthetic_labels = self._synthesize_labels(scaled, cluster_ids)
        rf = RandomForestClassifier(n_estimators=NUM_TREES, random_state=self.seed)
        rf.fit(scaled, synthetic_labels)
        predictions = rf.predict(scaled)
        accuracy = float(np.mean(predictions == synthetic_labels))

        risk_levels = {nid: str(level) for nid, level in zip(node_ids, predictions)}

        # Apply back to shared graph - this recomputes effective_cost everywhere.
        for nid, level in risk_levels.items():
            self.graph.set_crime_risk(nid, level)
        # Ensure cells we excluded still have a Low default (they may be empty).
        for n in self.graph.all_nodes():
            if n.id not in risk_levels:
                self.graph.set_crime_risk(n.id, "Low")

        # Police deployment - top NUM_POLICE risk scores. We use the model's
        # predicted probability of "High" as the ranking signal.
        police = self._deploy_police(rf, scaled, node_ids)

        return CrimeResult(
            cluster_labels=cluster_labels,
            risk_levels=risk_levels,
            police_deployment=police,
            accuracy=accuracy,
            cluster_centers=kmeans.cluster_centers_,
        )

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

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
            # No industrials - report a large constant so RF treats every node uniformly.
            self._industrial_distance_cache[node_id] = 99
            return 99

        # Multi-source BFS from all industrial cells - we only need to compute once
        # per pipeline run, but the cache is the cheap way to do it.
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

    # ------------------------------------------------------------------
    # Synthetic label generation
    # ------------------------------------------------------------------

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
        # Higher score = more risky.
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

        # Inject light noise so the model genuinely fits something non-trivial.
        for i in range(len(labels)):
            if rng.random() < SYNTHETIC_NOISE:
                labels[i] = rng.choice(list(RISK_MULTIPLIER.keys()))

        return labels

    # ------------------------------------------------------------------
    # Police deployment
    # ------------------------------------------------------------------

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
            # No "High" class in training data - fall back to first column.
            scores = proba[:, 0]

        # Bias slightly towards residential areas - that's where citizens live.
        for i, nid in enumerate(node_ids):
            if self.graph.node(nid).type == LOC_RESIDENTIAL:
                scores[i] += 0.05

        ranked = sorted(zip(node_ids, scores), key=lambda kv: kv[1], reverse=True)
        return [nid for nid, _ in ranked[:NUM_POLICE]]


def predict_crime_risk(graph: CityGraph, seed: Optional[int] = None) -> CrimeResult:
    """Convenience entry point for the simulation loop."""
    return CrimeRiskPipeline(graph, seed=seed).run()
