"""K-means team classifier on shirt colours."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from src.team_assignment.features import lab_to_bgr


class TeamClassifier:
    """Cluster shirt colours into two teams.

    Clusters are ordered by lightness (team 0 = darker kit) so results are
    deterministic across runs.
    """

    def __init__(self, n_teams: int = 2, seed: int = 0) -> None:
        self.n_teams = n_teams
        self.seed = seed
        self.centroids: np.ndarray | None = None

    def fit(self, features: np.ndarray, trim_factor: float = 2.5) -> TeamClassifier:
        """Fit on an ``(N, 3)`` array of Lab colours (one per crop).

        Crops of referees or goalkeepers that slipped in as "players" would pull
        the team colours towards them, so the fit is done twice: crops further
        than ``trim_factor`` x their cluster's median distance from its centre are dropped
        before the second fit.
        """
        features = np.asarray(features, dtype=float)
        if len(features) < self.n_teams:
            raise ValueError(f"Need at least {self.n_teams} shirt crops, got {len(features)}")
        km = KMeans(n_clusters=self.n_teams, n_init=10, random_state=self.seed).fit(features)
        dist = np.linalg.norm(features - km.cluster_centers_[km.labels_], axis=1)
        inliers = np.ones(len(features), dtype=bool)
        for k in range(self.n_teams):
            in_k = km.labels_ == k
            inliers[in_k] = dist[in_k] <= trim_factor * max(np.median(dist[in_k]), 5.0)
        if inliers.sum() >= self.n_teams and not inliers.all():
            km = KMeans(n_clusters=self.n_teams, n_init=10, random_state=self.seed).fit(features[inliers])
        self.centroids = km.cluster_centers_[np.argsort(km.cluster_centers_[:, 0])]
        return self

    def distances(self, features: np.ndarray) -> np.ndarray:
        """``(N, n_teams)`` Lab distances from each feature to each team colour."""
        if self.centroids is None:
            raise RuntimeError("TeamClassifier is not fitted")
        return np.linalg.norm(np.asarray(features)[:, None, :] - self.centroids[None], axis=2)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Team index (0..n_teams-1) of each feature."""
        return self.distances(features).argmin(axis=1)

    def colours_bgr(self) -> list[tuple[int, int, int]]:
        """Team colours as BGR tuples, for drawing."""
        if self.centroids is None:
            raise RuntimeError("TeamClassifier is not fitted")
        return [lab_to_bgr(c) for c in self.centroids]
