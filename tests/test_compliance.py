"""Unit Tests for Trajectory Compliance Engine & AI Classifier."""

import numpy as np
import pytest

from src.compliance.classifier import ComplianceMLP, TrajectoryClassifier
from src.compliance.feature_extractor import TrajectoryFeatureExtractor
from src.compliance.geometry import RoundaboutGeometry


def test_roundabout_geometry_zone_classification():
    """Verify spatial classification of central island, inner ring, and outer ring."""
    geom = RoundaboutGeometry(center=(0.0, 0.0), r_island=10.0, r_inner=15.0, r_outer=20.0)

    assert geom.classify_zone(0.0, 5.0) == "island"
    assert geom.classify_zone(0.0, 12.0) == "inner_ring"
    assert geom.classify_zone(0.0, 18.0) == "outer_ring"
    assert geom.classify_zone(0.0, 30.0) == "outside"


def test_trajectory_feature_extractor_dimension():
    """Verify feature vector length is exactly 74 dimensions."""
    geom = RoundaboutGeometry(center=(0.0, 0.0))
    extractor = TrajectoryFeatureExtractor(geom, num_resample_points=32)

    # Synthetic circle trajectory
    theta = np.linspace(0, np.pi, 20)
    pts = [(float(18.0 * np.cos(t)), float(18.0 * np.sin(t))) for t in theta]

    feat = extractor.extract_features(pts)
    assert feat["is_valid"] is True
    # 32 points * 2 coords + 10 scalars = 74 dimensions
    assert len(feat["vector"]) == 74
    assert "metrics" in feat


def test_compliance_rule_evaluation():
    """Verify Portuguese Highway Code compliance logic on synthetic trajectories."""
    geom = RoundaboutGeometry(center=(0.0, 0.0), r_island=10.0, r_inner=15.0, r_outer=20.0)

    # 1. Exit 1 maneuver in outer lane (compliant)
    theta_exit1 = np.linspace(0, np.pi / 2, 25)
    traj_exit1 = [(float(18.0 * np.cos(t)), float(18.0 * np.sin(t))) for t in theta_exit1]
    res_exit1 = geom.evaluate_compliance(traj_exit1, entry_leg=0, exit_leg=1)
    assert res_exit1["is_compliant"] is True
    assert res_exit1["label"] == "COMPLIANT_EXIT_1"

    # 2. Outer lane misuse (Exit 3 in outer lane) -> infraction
    theta_exit3 = np.linspace(0, 3 * np.pi / 2, 50)
    traj_outer_misuse = [(float(18.5 * np.cos(t)), float(18.5 * np.sin(t))) for t in theta_exit3]
    res_misuse = geom.evaluate_compliance(traj_outer_misuse, entry_leg=0, exit_leg=3)
    assert res_misuse["is_compliant"] is False
    assert res_misuse["label"] == "INFRACTION_OUTER_MISUSE"


def test_ai_classifier_inference():
    """Verify neural model / classifier inference wrapper returns valid probabilities."""
    classifier = TrajectoryClassifier(model_path="models/compliance_mlp.pt")
    theta = np.linspace(0, np.pi / 2, 30)
    traj = [(-6.4 + float(19.0 * np.cos(t)), 61.0 + float(19.0 * np.sin(t))) for t in theta]

    res = classifier.classify_trajectory(traj)
    assert "is_compliant" in res
    assert "label" in res
    assert "citation" in res
    assert res["label"] in classifier.CLASSES

