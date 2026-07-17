"""hqopt CLI 参数解析与分发测试（不实际跑优化/回测）。"""
import pytest

from hqopt.cli import _override_alpha_file, build_parser


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


def test_alpha_file_override_clears_default_synthetic_flag():
    cfg = {"alpha": {"source": "file", "path": "synthetic.parquet", "synthetic": True}}

    _override_alpha_file(cfg, "real_alpha.parquet")

    assert cfg["alpha"] == {
        "source": "file",
        "path": "real_alpha.parquet",
        "synthetic": False,
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


def test_data_invalid_choice():
    with pytest.raises(SystemExit):
        build_parser().parse_known_args(["data", "nope"])


def test_subcommand_required():
    with pytest.raises(SystemExit):
        build_parser().parse_known_args([])
