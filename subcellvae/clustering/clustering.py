"""
Clustering utilities for latent feature analysis.

Covers KMeans and DBSCAN fitting, model persistence, and label prediction.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
from sklearn.cluster import DBSCAN, KMeans


def kmeans_cluster(latents: np.ndarray, num_clusters: int,
                   result_dir: str, model_name: str):
    """
    Fit KMeans and save the model.

    Returns
    -------
    kmeans : fitted KMeans
    labels : (N,) np.ndarray
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(latents)
    joblib.dump(kmeans, os.path.join(result_dir, model_name + '.pkl'))
    return kmeans, kmeans.labels_


def kmeans_latents(model_path: str, latents: np.ndarray) -> np.ndarray:
    """Predict KMeans cluster labels using a saved model."""
    return joblib.load(model_path).predict(latents)


def DBSCAN_cluster(latents: np.ndarray, eps: float, min_samples: int,
                   result_dir: str, model_name: str):
    """
    Fit DBSCAN and save the model.

    Returns
    -------
    db     : fitted DBSCAN
    labels : (N,) np.ndarray  (-1 indicates noise)
    """
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(latents)
    joblib.dump(db, os.path.join(result_dir, model_name + '.pkl'))
    return db, db.labels_
