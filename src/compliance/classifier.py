"""AI Trajectory Compliance Classifier for Portuguese Highway Code (Art. 14.º-A).

Integrates a PyTorch Multi-Layer Perceptron (MLP) neural network and rule-based
geometric verification to classify roundabout vehicle trajectories into:
1. COMPLIANT_EXIT_1: Immediate exit from outer lane
2. COMPLIANT_EXIT_2PLUS: Inner ring circulation with proper exit transition
3. INFRACTION_OUTER_MISUSE: Circulating past exit 1 solely in outer lane
4. INFRACTION_DIRECT_CUTOFF: Cutting across from inner ring straight into exit
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

from src.compliance.feature_extractor import TrajectoryFeatureExtractor
from src.compliance.geometry import RoundaboutGeometry


if nn is not None:
    class ComplianceMLP(nn.Module):
        """Lightweight Deep MLP for real-time edge trajectory compliance classification."""

        def __init__(self, in_features: int = 74, num_classes: int = 4) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.20),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
else:
    ComplianceMLP = None  # type: ignore


class TrajectoryClassifier:
    """Production inference wrapper combining AI neural prediction and rule verification."""

    CLASSES = [
        "COMPLIANT_EXIT_1",
        "COMPLIANT_EXIT_2PLUS",
        "INFRACTION_OUTER_MISUSE",
        "INFRACTION_DIRECT_CUTOFF",
    ]

    PORTUGUESE_CODE_CITATIONS = {
        "COMPLIANT_EXIT_1": "Código da Estrada, Art. 14.º-A, n.º 1, al. a) (Conforme)",
        "COMPLIANT_EXIT_2PLUS": "Código da Estrada, Art. 14.º-A, n.º 1, al. b) e c) (Conforme)",
        "INFRACTION_OUTER_MISUSE": "Infração ao Art. 14.º-A, n.º 1 e 2: Utilização indevida da via mais à direita",
        "INFRACTION_DIRECT_CUTOFF": "Infração ao Art. 14.º-A, n.º 1, al. c): Saída direta da via interior sem transição prévia",
    }

    def __init__(
        self,
        model_path: Optional[str] = "models/compliance_mlp.pt",
        geometry: Optional[RoundaboutGeometry] = None,
        device: Optional[str] = None,
    ) -> None:
        self.geom = geometry or RoundaboutGeometry()
        self.feature_extractor = TrajectoryFeatureExtractor(self.geom)

        if device is None:
            self.device = "cuda:0" if (torch is not None and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device

        self.model: Optional[Any] = None
        if model_path is not None and Path(model_path).exists() and torch is not None:
            self.load_model(model_path)

    def load_model(self, path: str) -> None:
        """Load trained PyTorch weights from checkpoint file."""
        if torch is None or ComplianceMLP is None:
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model = ComplianceMLP(in_features=74, num_classes=len(self.CLASSES))
        self.model.load_state_dict(ckpt)
        self.model.to(self.device)
        self.model.eval()

    def classify_trajectory(
        self,
        trajectory: List[Tuple[float, float]],
        entry_leg: Optional[int] = None,
        exit_leg: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Classify a vehicle trajectory and return detailed compliance assessment."""
        # 1. Evaluate baseline geometric rules
        rule_eval = self.geom.evaluate_compliance(trajectory, entry_leg, exit_leg)

        # 2. Extract 74-dim normalized feature representation
        feat_data = self.feature_extractor.extract_features(trajectory, entry_leg, exit_leg)
        if not feat_data["is_valid"]:
            return rule_eval

        feat_vec = feat_data["vector"]
        metrics = feat_data["metrics"]

        # 3. If neural model is loaded, run forward pass
        ai_pred: Optional[str] = None
        probabilities: Dict[str, float] = {}

        if self.model is not None and torch is not None:
            with torch.no_grad():
                inp = torch.tensor(feat_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits = self.model(inp)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]

                best_idx = int(np.argmax(probs))
                ai_pred = self.CLASSES[best_idx]
                probabilities = {
                    cls_name: round(float(probs[i]), 4)
                    for i, cls_name in enumerate(self.CLASSES)
                }

        # 4. Synthesize final decision (AI prediction validated by Highway Code geometry)
        final_label = rule_eval["label"]
        if ai_pred is not None and probabilities.get(ai_pred, 0.0) >= 0.70:
            final_label = ai_pred

        is_compliant = "COMPLIANT" in final_label
        citation = self.PORTUGUESE_CODE_CITATIONS.get(final_label, "Código da Estrada, Art. 14.º-A")

        return {
            "is_compliant": is_compliant,
            "label": final_label,
            "citation": citation,
            "confidence": probabilities.get(final_label, rule_eval["confidence"]),
            "reason": rule_eval["reason"],
            "probabilities": probabilities,
            "metrics": metrics,
        }
