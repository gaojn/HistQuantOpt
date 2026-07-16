"""两个 default 配置可加载且字段完整。"""
from hqopt.pipeline.batch_optimize import load_config, _parse_style_bound


def test_default_configs_load():
    for path, strat in [
        ("configs/alpha_max_default.yaml", "alpha_max"),
        ("configs/index_enhance_default.yaml", "index_enhance"),
    ]:
        cfg = load_config(path)
        assert cfg["strategy"] == strat
        assert "execution" in cfg
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
