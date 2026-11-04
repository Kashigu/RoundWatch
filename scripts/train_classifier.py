"""Training and Validation Pipeline for Trajectory Compliance AI Classifier.

Generates balanced training data modeling Portuguese Highway Code (Art. 14.º-A)
roundabout maneuvers and infractions, trains a PyTorch deep MLP, evaluates
generalization metrics, and saves model weights to models/compliance_mlp.pt.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath("."))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.compliance.classifier import ComplianceMLP, TrajectoryClassifier
from src.compliance.feature_extractor import TrajectoryFeatureExtractor
from src.compliance.geometry import RoundaboutGeometry


def generate_synthetic_trajectory(
    label: str,
    geom: RoundaboutGeometry,
    num_samples: int = 50,
) -> Tuple[List[Tuple[float, float]], int, int]:
    """Generate a single trajectory matching the desired compliance/infraction class."""
    cx, cy = geom.cx, geom.cy
    r_island = geom.r_island
    r_inner = geom.r_inner
    r_outer = geom.r_outer

    # Pick entry leg: 0, 1, 2, or 3
    entry_leg = random.randint(0, 3)
    entry_ang = geom.leg_angles[entry_leg]

    points: List[Tuple[float, float]] = []

    if label == "COMPLIANT_EXIT_1":
        exit_leg = (entry_leg + 1) % 4
        # Approach in outer lane
        r_circ = r_outer - random.uniform(0.5, 1.8)
        # Angles traversed: ~90 degrees counter-clockwise
        span_deg = 90.0 + random.uniform(-5.0, 5.0)
        for i in range(num_samples):
            frac = i / float(num_samples - 1)
            ang = math.radians(entry_ang + frac * span_deg)
            r = r_circ + random.gauss(0, 0.15)
            # Add entry/exit straight extensions
            if frac < 0.15:
                dist_ext = (0.15 - frac) * 120.0
                px = cx + math.cos(math.radians(entry_ang)) * (r_outer + dist_ext)
                py = cy + math.sin(math.radians(entry_ang)) * (r_outer + dist_ext)
            elif frac > 0.85:
                dist_ext = (frac - 0.85) * 120.0
                exit_ang = entry_ang + span_deg
                px = cx + math.cos(math.radians(exit_ang)) * (r_outer + dist_ext)
                py = cy + math.sin(math.radians(exit_ang)) * (r_outer + dist_ext)
            else:
                px = cx + r * math.cos(ang)
                py = cy + r * math.sin(ang)
            points.append((float(px), float(py)))

    elif label == "COMPLIANT_EXIT_2PLUS":
        # Exit 2 or Exit 3
        num_exits = random.choice([2, 3])
        exit_leg = (entry_leg + num_exits) % 4
        span_deg = num_exits * 90.0 + random.uniform(-4.0, 4.0)

        # Circulate inner ring until immediately after prior exit (num_exits - 1) * 90
        shift_start_ang = (num_exits - 1) * 90.0 + random.uniform(5.0, 20.0)
        shift_end_ang = shift_start_ang + random.uniform(35.0, 50.0)

        r_in = r_inner - random.uniform(0.8, 2.0)
        r_out = r_outer - random.uniform(0.5, 1.8)

        for i in range(num_samples):
            frac = i / float(num_samples - 1)
            cur_span = frac * span_deg
            ang = math.radians(entry_ang + cur_span)

            if cur_span < shift_start_ang:
                r = r_in
            elif cur_span < shift_end_ang:
                t = (cur_span - shift_start_ang) / (shift_end_ang - shift_start_ang)
                r = r_in + t * (r_out - r_in)
            else:
                r = r_out

            r += random.gauss(0, 0.12)
            px = cx + r * math.cos(ang)
            py = cy + r * math.sin(ang)
            points.append((float(px), float(py)))

    elif label == "INFRACTION_OUTER_MISUSE":
        # Exit 2 or 3, but circulates continuously in the outer lane!
        num_exits = random.choice([2, 3])
        exit_leg = (entry_leg + num_exits) % 4
        span_deg = num_exits * 90.0 + random.uniform(-4.0, 4.0)

        r_out = r_outer - random.uniform(0.4, 1.5)
        for i in range(num_samples):
            frac = i / float(num_samples - 1)
            cur_span = frac * span_deg
            ang = math.radians(entry_ang + cur_span)
            r = r_out + random.gauss(0, 0.15)
            px = cx + r * math.cos(ang)
            py = cy + r * math.sin(ang)
            points.append((float(px), float(py)))

    elif label == "INFRACTION_DIRECT_CUTOFF":
        # Exit 2 or 3, remains in inner ring all the way and cuts diagonally straight across to exit!
        num_exits = random.choice([2, 3])
        exit_leg = (entry_leg + num_exits) % 4
        span_deg = num_exits * 90.0 + random.uniform(-4.0, 4.0)

        r_in = r_inner - random.uniform(0.8, 2.0)
        for i in range(num_samples):
            frac = i / float(num_samples - 1)
            cur_span = frac * span_deg
            ang = math.radians(entry_ang + cur_span)

            # Stays in inner ring until the very end (> 92% of path)
            if frac < 0.92:
                r = r_in
            else:
                t = (frac - 0.92) / 0.08
                r = r_in + t * (r_outer - r_in + 2.0)

            r += random.gauss(0, 0.12)
            px = cx + r * math.cos(ang)
            py = cy + r * math.sin(ang)
            points.append((float(px), float(py)))

    else:
        raise ValueError(f"Unknown label: {label}")

    return points, entry_leg, exit_leg


def build_dataset(
    num_samples_per_class: int = 300,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build feature dataset across all 4 Highway Code classes."""
    geom = RoundaboutGeometry()
    extractor = TrajectoryFeatureExtractor(geom)
    classes = TrajectoryClassifier.CLASSES

    X_list = []
    y_list = []

    for cls_idx, cls_name in enumerate(classes):
        for _ in range(num_samples_per_class):
            pts, e_leg, x_leg = generate_synthetic_trajectory(cls_name, geom)
            feat = extractor.extract_features(pts, e_leg, x_leg)
            if feat["is_valid"]:
                X_list.append(feat["vector"])
                y_list.append(cls_idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, classes


def train_model() -> Dict[str, Any]:
    """Train PyTorch MLP model on trajectory dataset and evaluate performance."""
    print("[AI Classifier] Generating dataset (350 trajectories per class)...")
    X, y, classes = build_dataset(num_samples_per_class=350)
    print(f"[AI Classifier] Dataset assembled: {X.shape[0]} total samples, {X.shape[1]} features.")

    # Train / Test split (80 / 20)
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)

    split = int(0.80 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, y_train = torch.tensor(X[train_idx]), torch.tensor(y[train_idx])
    X_test, y_test = torch.tensor(X[test_idx]), torch.tensor(y[test_idx])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AI Classifier] Training on compute device: {device}")

    model = ComplianceMLP(in_features=X.shape[1], num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    best_acc = 0.0
    for epoch in range(1, 31):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                preds = torch.argmax(model(bx), dim=1)
                correct += (preds == by).sum().item()
                total += len(by)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/compliance_mlp.pt")

        if epoch % 5 == 0 or epoch == 30:
            avg_loss = total_loss / len(train_idx)
            print(f"  Epoch {epoch:2d}/30 | Loss: {avg_loss:.4f} | Test Acc: {acc:.2%}")

    # Final detailed evaluation & Confusion Matrix
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            preds = torch.argmax(model(bx), dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(by.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Per-class metrics
    metrics_per_class = {}
    for idx, name in enumerate(classes):
        tp = np.sum((all_preds == idx) & (all_targets == idx))
        fp = np.sum((all_preds == idx) & (all_targets != idx))
        fn = np.sum((all_preds != idx) & (all_targets == idx))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        metrics_per_class[name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "samples": int(np.sum(all_targets == idx)),
        }

    overall_acc = float(np.mean(all_preds == all_targets))
    overall_f1 = float(np.mean([m["f1_score"] for m in metrics_per_class.values()]))

    results = {
        "overall_accuracy": round(overall_acc, 4),
        "macro_f1": round(overall_f1, 4),
        "best_epoch_acc": round(best_acc, 4),
        "per_class": metrics_per_class,
        "classes": classes,
    }

    with open("models/classifier_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("AI COMPLIANCE CLASSIFIER EVALUATION REPORT")
    print("=" * 60)
    print(f"Overall Test Accuracy: {overall_acc:.2%}")
    print(f"Macro F1-Score:        {overall_f1:.4f}")
    print("-" * 60)
    for name, m in metrics_per_class.items():
        print(f"  {name:26s} | Prec: {m['precision']:.3f} | Rec: {m['recall']:.3f} | F1: {m['f1_score']:.3f} (N={m['samples']})")
    print("=" * 60)
    print("[AI Classifier] Weights saved to models/compliance_mlp.pt")
    return results


if __name__ == "__main__":
    train_model()

