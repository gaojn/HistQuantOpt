"""两个 default 配置可加载且字段完整。"""
from pathlib import Path

import pytest

from hqopt.pipeline.batch.config import (
    _synthetic_alpha_enabled,
    normalize_config_paths,
)
from hqopt.pipeline.batch_optimize import _parse_style_bound, load_config


def test_default_configs_load():
    for path, strat in [
        ("configs/alpha_max_default.yaml", "alpha_max"),
        ("configs/index_enhance_default.yaml", "index_enhance"),
    ]:
        cfg = load_config(path)
        assert cfg["strategy"] == strat
        assert Path(cfg["data"]["root"]) == Path.cwd().resolve()
        assert "execution" in cfg
        assert cfg["alpha"]["synthetic"] is True
        assert cfg["alpha"]["max_staleness_days"] == 15
        if strat == "index_enhance":
            assert cfg["optimizer"]["benchmark_max_snapshot_age_days"] == 30
        else:
            assert cfg["backtest"]["benchmark"] == "equal_weight"
        for k in ("start_date", "end_date", "rebalance_freq", "initial_value"):
            assert k in cfg["backtest"]
        assert "weight_upper" in cfg["optimizer"]


def test_index_enhance_default_style_dict_parses():
    cfg = load_config("configs/index_enhance_default.yaml")
    b = _parse_style_bound(cfg["optimizer"]["style_active_bound"])
    assert isinstance(b, dict)
    assert b["Momentum"] == 0.20


def test_alpha_max_default_style_dict_parses():
    cfg = load_config("configs/alpha_max_default.yaml")
    b = _parse_style_bound(cfg["optimizer"]["style_bound"])
    assert isinstance(b, dict)
    assert b["Size"] == 0.20


def test_file_alpha_requires_explicit_synthetic_declaration():
    with pytest.raises(ValueError, match="必须显式设置 alpha.synthetic"):
        _synthetic_alpha_enabled({"source": "file", "path": "alpha.parquet"})


@pytest.mark.parametrize(
    ("alpha_cfg", "message"),
    [
        ({}, "缺少 alpha.source"),
        (
            {"source": "unknown", "synthetic": False},
            "alpha.source 须为",
        ),
        (
            {"source": "file", "synthetic": "false"},
            "alpha.synthetic 必须是布尔值",
        ),
        (
            {"source": "synthetic", "synthetic": False},
            "与 alpha.synthetic=false 矛盾",
        ),
    ],
)
def test_invalid_alpha_metadata_is_rejected(alpha_cfg, message):
    with pytest.raises(ValueError, match=message):
        _synthetic_alpha_enabled(alpha_cfg)


def test_valid_alpha_metadata_returns_synthetic_flag():
    assert _synthetic_alpha_enabled(
        {"source": "file", "synthetic": False}
    ) is False
    assert _synthetic_alpha_enabled(
        {"source": "file", "synthetic": True}
    ) is True
    assert _synthetic_alpha_enabled({"source": "synthetic"}) is True


def test_config_paths_resolve_from_external_data_root(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "run.yaml"
    config_path.write_text(
        """
strategy: alpha_max
data:
  root: ../bundle
optimizer:
  cne6_data_dir: data/barra_cne6_L
alpha:
  source: file
  synthetic: false
  path: alphas/signal.parquet
output:
  weights: output/weights.parquet
""".lstrip(),
        encoding="utf-8",
    )

    cfg = load_config(config_path)
    root = (tmp_path / "bundle").resolve()

    assert Path(cfg["data"]["root"]) == root
    assert Path(cfg["optimizer"]["cne6_data_dir"]) == (
        root / "data" / "barra_cne6_L"
    )
    assert Path(cfg["alpha"]["path"]) == root / "alphas" / "signal.parquet"
    assert Path(cfg["output"]["weights"]) == root / "output" / "weights.parquet"


def test_normalize_config_paths_does_not_mutate_caller(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = {
        "data": {"root": "bundle"},
        "optimizer": {},
        "alpha": {"source": "synthetic"},
        "output": {"weights": "output/weights.parquet"},
    }

    normalized = normalize_config_paths(raw)

    assert raw["data"]["root"] == "bundle"
    assert raw["output"]["weights"] == "output/weights.parquet"
    assert normalized["data"]["root"] == str((tmp_path / "bundle").resolve())
