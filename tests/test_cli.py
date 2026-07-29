"""hqopt CLI 参数解析与分发测试（不实际跑优化/回测）。"""
import json
from pathlib import Path

import pytest

import hqopt.analysis.run as attribution_run
import hqopt.backtest.run as backtest_run
from hqopt.cli import (
    ROOT,
    _default_attribution_out_dir,
    _default_backtest_out_dir,
    _override_alpha_file,
    build_parser,
    cmd_attribute,
    cmd_backtest,
)
from hqopt.io import run_manifest


def test_run_parses():
    args, extra = build_parser().parse_known_args(["run", "configs/x.yaml"])
    assert args.cmd == "run"
    assert args.config == "configs/x.yaml"
    assert extra == []


def test_optimize_overrides():
    args, _ = build_parser().parse_known_args(
        ["optimize", "c.yaml", "--risk-aversion", "12", "--output", "w.parquet"]
    )
    assert args.cmd == "optimize"
    assert args.risk_aversion == 12.0
    assert args.output == "w.parquet"


def test_alpha_file_override_preserves_synthetic_declaration():
    cfg = {"alpha": {"source": "file", "path": "synthetic.parquet", "synthetic": True}}

    _override_alpha_file(cfg, "real_alpha.parquet")

    assert cfg["alpha"] == {
        "source": "file",
        "path": str(Path("real_alpha.parquet").resolve()),
        "synthetic": True,
    }


def test_alpha_file_override_preserves_real_declaration():
    cfg = {"alpha": {"source": "file", "path": "old.parquet", "synthetic": False}}

    _override_alpha_file(cfg, "new.parquet")

    assert cfg["alpha"]["path"] == str(Path("new.parquet").resolve())
    assert cfg["alpha"]["synthetic"] is False


def test_alpha_file_override_freezes_synthetic_source_as_watermark():
    cfg = {"alpha": {"source": "synthetic"}}

    _override_alpha_file(cfg, "exported_synthetic.parquet")

    assert cfg["alpha"] == {
        "source": "file",
        "path": str(Path("exported_synthetic.parquet").resolve()),
        "synthetic": True,
    }


def test_backtest_requires_weights():
    with pytest.raises(SystemExit):
        build_parser().parse_known_args(["backtest", "--start", "2020-01-01", "--end", "2026-01-01"])


def test_backtest_full():
    args, _ = build_parser().parse_known_args(
        ["backtest", "--weights", "w.parquet", "--start", "2020-01-01",
         "--end", "2026-05-31", "--index", "zz500"]
    )
    assert args.weights == "w.parquet" and args.index == "zz500"


def test_backtest_risk_free_help_matches_default_behavior():
    parser = build_parser()
    backtest_parser = parser._subparsers._group_actions[0].choices["backtest"]
    help_text = backtest_parser.format_help()

    assert "默认 0.02" in help_text
    assert "YAML" not in help_text


def test_backtest_default_output_stays_with_standard_strategy():
    out_dir = _default_backtest_out_dir(
        "output/index_enhance_default/weights.parquet"
    )

    assert out_dir == (ROOT / "output" / "index_enhance_default").resolve()


def test_backtest_external_weights_default_to_project_output(tmp_path):
    out_dir = _default_backtest_out_dir(tmp_path / "weights.parquet")

    assert out_dir == (ROOT / "output" / "backtest").resolve()


def test_data_passthrough():
    args, extra = build_parser().parse_known_args(["data", "index-close", "--start", "2015-01-01"])
    assert args.cmd == "data" and args.what == "index-close"
    assert extra == ["--start", "2015-01-01"]


def test_attribute_execution_options():
    args, _ = build_parser().parse_known_args([
        "attribute", "--weights", "w.parquet", "--start", "2024-01-01",
        "--end", "2024-12-31", "--cost-buy", "0.002", "--initial-value", "50000000",
    ])
    assert args.cost_buy == 0.002
    assert args.initial_value == 5e7


def test_attribute_default_output_stays_with_standard_strategy():
    out_dir = _default_attribution_out_dir(
        "output/index_enhance_default/weights.parquet"
    )

    assert out_dir == (
        ROOT / "output" / "index_enhance_default" / "attribution"
    ).resolve()


def test_attribute_external_weights_default_to_project_output(tmp_path):
    out_dir = _default_attribution_out_dir(tmp_path / "weights.parquet")

    assert out_dir == (ROOT / "output" / "attribution").resolve()


def test_data_invalid_choice():
    with pytest.raises(SystemExit):
        build_parser().parse_known_args(["data", "nope"])


def test_subcommand_required():
    with pytest.raises(SystemExit):
        build_parser().parse_known_args([])


def test_attribute_benchmark_weight_source_defaults_to_drift():
    args = build_parser().parse_args([
        "attribute", "--weights", "w.parquet", "--start", "2024-01-01",
        "--end", "2024-12-31",
    ])
    assert args.benchmark_weight_source == "official_drift"
    assert args.benchmark_max_snapshot_age_days == 30


def test_attribute_accepts_legacy_frozen_benchmark_weights():
    args = build_parser().parse_args([
        "attribute", "--weights", "w.parquet", "--start", "2024-01-01",
        "--end", "2024-12-31", "--benchmark-weight-source", "official_frozen",
    ])
    assert args.benchmark_weight_source == "official_frozen"


def _write_artifacts(output_dir: Path, relative_paths: list[str]) -> None:
    for relative in relative_paths:
        path = output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))


def _stable_repository_state(root: Path) -> dict:
    return {
        "root": str(root),
        "commit": "abc123",
        "branch": "main",
        "dirty": False,
        "tracked_diff_sha256": "d" * 64,
        "worktree_sha256": "e" * 64,
        "untracked_files": [],
    }


def test_independent_backtest_writes_complete_run_manifest(
    tmp_path,
    monkeypatch,
):
    weights = tmp_path / "weights.parquet"
    weights.write_bytes(b"weights")
    output_dir = tmp_path / "backtest"
    monkeypatch.setattr(
        run_manifest,
        "collect_repository_state",
        _stable_repository_state,
    )

    def fake_run_backtest(*args, **kwargs):
        del args
        out = Path(kwargs["out_dir"])
        _write_artifacts(
            out,
            [
                "report.html",
                "nav.parquet",
                "turnover.parquet",
                "actual_weights.parquet",
                "execution_stats.json",
                "report_data/timeseries.parquet",
                "report_data/turnover.parquet",
                "report_data/metrics.parquet",
                "report_data/yearly.parquet",
                "report_data/monthly_excess.parquet",
            ],
        )

    monkeypatch.setattr(backtest_run, "run_backtest", fake_run_backtest)
    args = build_parser().parse_args([
        "backtest",
        "--weights",
        str(weights),
        "--start",
        "2024-01-02",
        "--end",
        "2024-01-31",
        "--out-dir",
        str(output_dir),
    ])

    cmd_backtest(args)

    payload = json.loads(
        (output_dir / "run.manifest.json").read_text(encoding="utf-8")
    )
    assert payload["mode"] == "backtest"
    assert payload["status"] == "complete"
    assert payload["data_lock"]["verified"] is False
    assert payload["config"]["effective"]["risk_free"] == 0.02
    assert payload["config"]["effective"]["output_dir"] == str(
        output_dir.resolve()
    )
    assert {item["role"] for item in payload["inputs"]} == {"weights"}
    assert len(payload["artifacts"]) == 10
    assert not list(output_dir.glob(".run.*.in_progress.json"))


def test_independent_attribute_writes_complete_run_manifest(
    tmp_path,
    monkeypatch,
):
    weights = tmp_path / "weights.parquet"
    weights.write_bytes(b"weights")
    output_dir = tmp_path / "attribution"
    monkeypatch.setattr(
        run_manifest,
        "collect_repository_state",
        _stable_repository_state,
    )

    def fake_run_attribution(*args, **kwargs):
        del args
        _write_artifacts(
            Path(kwargs["out_dir"]),
            [
                "attribution_report.html",
                "attribution_summary.csv",
                "attribution_daily.parquet",
                "attribution_factor_daily.parquet",
                "attribution_execution_stats.json",
            ],
        )

    monkeypatch.setattr(
        attribution_run,
        "run_attribution",
        fake_run_attribution,
    )
    args = build_parser().parse_args([
        "attribute",
        "--weights",
        str(weights),
        "--start",
        "2024-01-02",
        "--end",
        "2024-01-31",
        "--out-dir",
        str(output_dir),
    ])

    cmd_attribute(args)

    payload = json.loads(
        (output_dir / "run.manifest.json").read_text(encoding="utf-8")
    )
    assert payload["mode"] == "attribute"
    assert payload["status"] == "complete"
    assert {item["role"] for item in payload["inputs"]} == {"weights"}
    assert len(payload["artifacts"]) == 5
    assert not list(output_dir.glob(".run.*.in_progress.json"))


def test_independent_backtest_failure_writes_failed_run_manifest(
    tmp_path,
    monkeypatch,
):
    weights = tmp_path / "weights.parquet"
    weights.write_bytes(b"weights")
    output_dir = tmp_path / "backtest"
    monkeypatch.setattr(
        run_manifest,
        "collect_repository_state",
        _stable_repository_state,
    )

    def fail_backtest(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("simulated backtest failure")

    monkeypatch.setattr(backtest_run, "run_backtest", fail_backtest)
    args = build_parser().parse_args([
        "backtest",
        "--weights",
        str(weights),
        "--start",
        "2024-01-02",
        "--end",
        "2024-01-31",
        "--out-dir",
        str(output_dir),
    ])

    with pytest.raises(RuntimeError, match="simulated backtest failure"):
        cmd_backtest(args)

    payload = json.loads(
        (output_dir / "run.manifest.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "failed"
    assert payload["error"] == "RuntimeError: simulated backtest failure"
    assert payload["artifacts"] == []
    assert not list(output_dir.glob(".run.*.in_progress.json"))
