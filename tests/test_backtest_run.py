from hqopt.backtest.run import _load_warning_banner
from hqopt.constants import SYNTHETIC_ALPHA_WARNING_FILE


def test_load_warning_banner_from_weights_directory(tmp_path):
    weights = tmp_path / "weights.parquet"
    assert _load_warning_banner(weights) is None

    warning = tmp_path / SYNTHETIC_ALPHA_WARNING_FILE
    warning.write_text("含未来信息，禁止用于实盘。\n", encoding="utf-8")

    assert _load_warning_banner(weights) == "含未来信息，禁止用于实盘。"
