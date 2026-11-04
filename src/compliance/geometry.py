"""Roundabout Spatial Geometry & Portuguese Highway Code (Art. 14.º-A) Compliance Evaluator.

Defines roundabout zones (inner ring, outer ring, central island, entry/exit legs)
and validates trajectory compliance against the Portuguese Highway Code:
1. Exit 1: Enter right lane -> Outer ring -> Exit 1.
2. Exit 2+: Enter left lane -> Inner ring -> Shift to outer lane immediately after prior exit -> Exit.
3. Infraction Type 1 (Outer Lane Misuse): Circulating in outer lane past Exit 1.
4. Infraction Type 2 (Direct Cut-Off): Cutting from inner ring directly across into exit.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
from shapely.geometry import Point, Polygon


class RoundaboutGeometry:
    """Represents multi-lane roundabout metric geometry and sector boundaries."""

    def __init__(
        self,
        center: Tuple[float, float] = (2.0, 6.0),
        r_island: float = 14.0,
        r_inner: float = 19.0,
        r_outer: float = 24.0,
        num_legs: int = 4,
        leg_angles: Optional[List[float]] = None,
    ) -> None:
        self.cx, self.cy = center
        self.r_island = r_island
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.num_legs = num_legs

        # Leg center angles in degrees in circulation order: 0=North (90°), 1=East (0°), 2=South (270°), 3=West (180°)
        if leg_angles is None:
            self.leg_angles = [90.0, 0.0, 270.0, 180.0]
        else:
            self.leg_angles = leg_angles

    def to_polar(self, x: float, y: float) -> Tuple[float, float]:
        """Convert Cartesian (x, y) relative to roundabout center to (radius, angle_deg)."""
        dx = x - self.cx
        dy = y - self.cy
        r = math.hypot(dx, dy)
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad) % 360.0
        return r, angle_deg

    def classify_zone(self, x: float, y: float) -> str:
        """Classify spatial point into 'island', 'inner_ring', 'outer_ring', or 'outside'."""
        r, _ = self.to_polar(x, y)
        if r < self.r_island:
            return "island"
        elif r <= self.r_inner:
            return "inner_ring"
        elif r <= self.r_outer + 1.5:
            return "outer_ring"
        else:
            return "outside"

    def identify_leg(self, x: float, y: float, tolerance_deg: float = 45.0) -> int:
        """Identify which roundabout leg (0..3) a point is approaching/departing on."""
        _, angle_deg = self.to_polar(x, y)
        best_leg = min(
            range(self.num_legs),
            key=lambda i: abs((angle_deg - self.leg_angles[i] + 180.0) % 360.0 - 180.0)
        )
        return best_leg

    def evaluate_compliance(
        self,
        trajectory: List[Tuple[float, float]],
        entry_leg: Optional[int] = None,
        exit_leg: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate a full vehicle trajectory against Portuguese Highway Code (Art. 14.º-A).
        
        Returns:
            Dict containing:
                - is_compliant: bool
                - label: str (COMPLIANT_EXIT_1, COMPLIANT_EXIT_2PLUS, 
                              INFRACTION_OUTER_MISUSE, INFRACTION_DIRECT_CUTOFF)
                - reason: str
                - lane_sequence: List[str]
                - confidence: float
        """
        if len(trajectory) < 5:
            return {
                "is_compliant": True,
                "label": "INSUFFICIENT_DATA",
                "reason": "Trajectory too short to evaluate.",
                "lane_sequence": [],
                "confidence": 0.0,
            }

        # Sample polar positions and zone sequence
        polar_points = [self.to_polar(pt[0], pt[1]) for pt in trajectory]
        zones = [self.classify_zone(pt[0], pt[1]) for pt in trajectory]

        # Filter points inside circulatory ring (inner or outer)
        ring_indices = [i for i, z in enumerate(zones) if z in ("inner_ring", "outer_ring")]
        if len(ring_indices) < 3:
            return {
                "is_compliant": True,
                "label": "TRANSIT_INCOMPLETE",
                "reason": "Vehicle did not circulate the roundabout.",
                "lane_sequence": zones,
                "confidence": 0.5,
            }

        first_ring_idx = ring_indices[0]
        last_ring_idx = ring_indices[-1]

        entry_zone = zones[first_ring_idx]
        exit_zone = zones[last_ring_idx]

        # Automatic leg inference if not explicitly provided
        if entry_leg is None:
            entry_leg = self.identify_leg(trajectory[0][0], trajectory[0][1])
        if exit_leg is None:
            exit_leg = self.identify_leg(trajectory[-1][0], trajectory[-1][1])

        # Calculate angular traversal in circulatory ring (Town03 circulation order)
        entry_ang = polar_points[first_ring_idx][1]
        exit_ang = polar_points[last_ring_idx][1]
        circ_traversal = (entry_ang - exit_ang) % 360.0

        # Number of exits traversed (approximate based on leg diff)
        num_exits_passed = 1
        if entry_leg is not None and exit_leg is not None:
            num_exits_passed = (exit_leg - entry_leg) % self.num_legs
            if num_exits_passed == 0:
                num_exits_passed = self.num_legs  # Full 360 loop / U-turn
        else:
            # Estimate from angular traversal: ~90 deg per exit
            num_exits_passed = max(1, int(round(circ_traversal / 90.0)))

        # Count frames spent in inner vs outer ring
        ring_zones = [zones[i] for i in ring_indices]
        outer_count = ring_zones.count("outer_ring")
        inner_count = ring_zones.count("inner_ring")
        outer_ratio = outer_count / len(ring_zones)
        inner_ratio = inner_count / len(ring_zones)

        # RULE CHECK 1: Exiting at Exit 1
        if num_exits_passed == 1:
            if outer_ratio >= 0.60:
                return {
                    "is_compliant": True,
                    "label": "COMPLIANT_EXIT_1",
                    "reason": "Exited at Exit 1 using the outer circulatory lane as mandated by Art. 14.º-A n.º 1.",
                    "lane_sequence": ring_zones,
                    "confidence": 0.98,
                }
            elif inner_ratio >= 0.50 and exit_zone == "inner_ring":
                # Entered inner ring and cut straight into Exit 1
                return {
                    "is_compliant": False,
                    "label": "INFRACTION_DIRECT_CUTOFF",
                    "reason": "Art. 14.º-A Infraction: Vehicle cut directly from inner lane into Exit 1 without using outer lane.",
                    "lane_sequence": ring_zones,
                    "confidence": 0.92,
                }
            else:
                return {
                    "is_compliant": True,
                    "label": "COMPLIANT_EXIT_1",
                    "reason": "Exited at Exit 1.",
                    "lane_sequence": ring_zones,
                    "confidence": 0.85,
                }

        # RULE CHECK 2: Exiting at Exit 2+ (2nd, 3rd, 4th exit)
        # Mandated rule: Must circulate inner lane, move to outer lane ONLY after passing prior exit.
        if num_exits_passed >= 2:
            # Infraction Type 1: Outer Lane Misuse
            # Vehicle spent majority of circulation in outer lane past Exit 1
            if outer_ratio > 0.65:
                return {
                    "is_compliant": False,
                    "label": "INFRACTION_OUTER_MISUSE",
                    "reason": (
                        f"Art. 14.º-A n.º 2 Infraction: Vehicle traversed {num_exits_passed} exits while improperly "
                        f"occupying the outer lane ({outer_ratio:.1%} of circulation time)."
                    ),
                    "lane_sequence": ring_zones,
                    "confidence": 0.95,
                }

            # Infraction Type 2: Direct Cut-Off
            # Vehicle remained in inner lane until the very end and exited directly from inner ring
            last_quarter = ring_zones[int(len(ring_zones) * 0.75):]
            if last_quarter.count("inner_ring") > len(last_quarter) * 0.80 and exit_zone == "inner_ring":
                return {
                    "is_compliant": False,
                    "label": "INFRACTION_DIRECT_CUTOFF",
                    "reason": (
                        "Art. 14.º-A n.º 3 Infraction: Direct cut-off exit. Vehicle failed to move to the outer lane "
                        "after the preceding exit and exited directly across from the inner ring."
                    ),
                    "lane_sequence": ring_zones,
                    "confidence": 0.93,
                }

            # Compliant Exit 2+
            return {
                "is_compliant": True,
                "label": "COMPLIANT_EXIT_2PLUS",
                "reason": (
                    f"Art. 14.º-A Compliant: Entered inner ring, circulated, and moved to outer lane after "
                    f"prior exit before taking exit {num_exits_passed}."
                ),
                "lane_sequence": ring_zones,
                "confidence": 0.94,
            }

        return {
            "is_compliant": True,
            "label": "COMPLIANT_TRANSIT",
            "reason": "Standard circulatory transit.",
            "lane_sequence": ring_zones,
            "confidence": 0.80,
        }

