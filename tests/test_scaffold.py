"""Test suite for repository structure and configurations."""

from pathlib import Path
import yaml
import pytest


def test_configs_exist_and_are_valid():
    default_cfg = Path("configs/default.yaml")
    roundabout_cfg = Path("configs/roundabout.yaml")

    assert default_cfg.exists(), "configs/default.yaml missing"
    assert roundabout_cfg.exists(), "configs/roundabout.yaml missing"

    with open(default_cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        assert "simulation" in cfg
        assert "camera" in cfg
        assert "perception" in cfg
        assert "tracking" in cfg
        assert "compliance" in cfg

    with open(roundabout_cfg, "r", encoding="utf-8") as f:
        r_cfg = yaml.safe_load(f)
        assert "roundabout" in r_cfg
        assert "rules" in r_cfg
        assert "geometry" in r_cfg["roundabout"]
        assert len(r_cfg["roundabout"]["legs"]) == 4


def test_package_imports():
    import src
    import src.simulation
    import src.perception
    import src.compliance
    import src.visualization
    import src.evaluation

    assert src.__version__ == "0.1.0"

