"""Cluster-aware bootstrap helpers used in the dissertation analyses.

Questions from the same dialogue or story are related. The bootstrap therefore
resamples whole clusters instead of treating every question as independent.
"""

import numpy as np


def clustered_accuracy_bootstrap(
    rows,
    prediction_key,
    reference_key,
    cluster_key,
    repetitions=5000,
    seed=42,
):
    """Estimate accuracy and a percentile confidence interval by cluster."""
    rng = np.random.default_rng(seed)

    clusters = {}
    for row in rows:
        clusters.setdefault(row[cluster_key], []).append(row)

    cluster_ids = list(clusters)
    observed = _accuracy(rows, prediction_key, reference_key)

    bootstrap_scores = []

    for _ in range(repetitions):
        sampled_ids = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        sampled_rows = []

        for cluster_id in sampled_ids:
            sampled_rows.extend(clusters[cluster_id])

        bootstrap_scores.append(
            _accuracy(sampled_rows, prediction_key, reference_key)
        )

    low, high = np.percentile(bootstrap_scores, [2.5, 97.5])

    return {
        "estimate": observed,
        "ci_low": float(low),
        "ci_high": float(high),
    }


def paired_cluster_bootstrap(
    rows,
    first_key,
    second_key,
    reference_key,
    cluster_key,
    repetitions=5000,
    seed=42,
):
    """Estimate the paired accuracy difference between two systems by cluster."""
    rng = np.random.default_rng(seed)

    clusters = {}
    for row in rows:
        clusters.setdefault(row[cluster_key], []).append(row)

    cluster_ids = list(clusters)

    observed = (
        _accuracy(rows, first_key, reference_key)
        - _accuracy(rows, second_key, reference_key)
    )

    differences = []

    for _ in range(repetitions):
        sampled_ids = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        sampled_rows = []

        for cluster_id in sampled_ids:
            sampled_rows.extend(clusters[cluster_id])

        difference = (
            _accuracy(sampled_rows, first_key, reference_key)
            - _accuracy(sampled_rows, second_key, reference_key)
        )
        differences.append(difference)

    low, high = np.percentile(differences, [2.5, 97.5])

    return {
        "difference": observed,
        "ci_low": float(low),
        "ci_high": float(high),
    }


def _accuracy(rows, prediction_key, reference_key):
    if not rows:
        raise ValueError("Cannot calculate accuracy from an empty list.")

    correct = sum(
        row[prediction_key] == row[reference_key]
        for row in rows
    )

    return correct / len(rows)
