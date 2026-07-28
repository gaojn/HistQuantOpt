import json
import multiprocessing
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

import hqopt.backtest.execution as execution
import hqopt.backtest.run as runmod


def _frames(*, shifted: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.to_datetime(["2024-01-02", "2024-01-12"])
    weights = pd.DataFrame(
        {
            "A": [0.6, 0.4] if not shifted else [0.5, 0.3],
            "B": [0.4, 0.6] if not shifted else [0.5, 0.7],
        },
        index=index,
    )
    weights.index.name = "date"
    sell_only = pd.DataFrame(
        {
            "A": [False, True] if not shifted else [True, False],
            "B": [False, False] if not shifted else [False, True],
        },
        index=index,
    )
    sell_only.index.name = "date"
    return weights, sell_only


def _stats(*, shifted: bool = False) -> dict[str, object]:
    return {
        "expired_order_count": 1 if shifted else 0,
        "expired_notional": 100.0 if shifted else 0.0,
        "target_pending": shifted,
        "final_cash": 10.0 if shifted else 0.0,
        "final_nav": 1_000.0,
        "final_shares": {"A": 30 if shifted else 60, "B": 70 if shifted else 40},
        "order_states": {"A": "filled", "B": "pending_buy" if shifted else "filled"},
    }


def _publish(
    weight_path: Path,
    *,
    shifted: bool = False,
) -> tuple[Path, Path, Path, Path]:
    weights, sell_only = _frames(shifted=shifted)
    return execution.publish_batch_bundle(
        weights,
        sell_only,
        _stats(shifted=shifted),
        weight_path,
    )


def _holding_reader_worker(weight_path, ready, release, results):
    original_metadata_loader = runmod._load_sell_only_metadata

    def holding_metadata_loader(path):
        metadata = original_metadata_loader(path)
        ready.set()
        if not release.wait(15):
            raise TimeoutError("reader release timed out")
        return metadata

    runmod._load_sell_only_metadata = holding_metadata_loader
    try:
        weights, metadata = runmod._load_execution_bundle(weight_path)
        results.put(
            (
                "ok",
                float(weights.iloc[0]["A"]),
                bool(metadata.iloc[0]["A"]),
            )
        )
    except BaseException as exc:
        ready.set()
        results.put(("error", type(exc).__name__, str(exc)))


def _signaled_publisher_worker(
    weight_path,
    shifted,
    attempting,
    acquired,
    results,
):
    original_lock = execution.bundle_io_lock

    @contextmanager
    def signaled_lock(path, *, exclusive):
        if exclusive:
            attempting.set()
        with original_lock(path, exclusive=exclusive):
            if exclusive:
                acquired.set()
            yield

    execution.bundle_io_lock = signaled_lock
    try:
        _publish(weight_path, shifted=shifted)
        results.put(("ok",))
    except BaseException as exc:
        results.put(("error", type(exc).__name__, str(exc)))


def _mid_publish_worker(weight_path, shifted, paused, release, results):
    original_replace = Path.replace
    sidecar = execution.sell_only_path_for_weights(weight_path)

    def pause_after_sidecar_replace(self, target):
        replaced = original_replace(self, target)
        if Path(target) == sidecar:
            paused.set()
            if not release.wait(15):
                raise TimeoutError("publisher release timed out")
        return replaced

    Path.replace = pause_after_sidecar_replace
    try:
        _publish(weight_path, shifted=shifted)
        results.put(("ok",))
    except BaseException as exc:
        paused.set()
        results.put(("error", type(exc).__name__, str(exc)))
    finally:
        Path.replace = original_replace


def _signaled_reader_worker(weight_path, attempting, acquired, results):
    original_lock = runmod.bundle_io_lock

    @contextmanager
    def signaled_lock(path, *, exclusive):
        attempting.set()
        with original_lock(path, exclusive=exclusive):
            acquired.set()
            yield

    runmod.bundle_io_lock = signaled_lock
    try:
        weights, metadata = runmod._load_execution_bundle(weight_path)
        results.put(
            (
                "ok",
                float(weights.iloc[0]["A"]),
                bool(metadata.iloc[0]["A"]),
            )
        )
    except BaseException as exc:
        acquired.set()
        results.put(("error", type(exc).__name__, str(exc)))


def _assert_process_ok(process, results):
    process.join(20)
    if process.is_alive():
        process.terminate()
        process.join(5)
        pytest.fail(f"子进程未按时结束：{process.name}")
    assert process.exitcode == 0
    result = results.get(timeout=2)
    assert result[0] == "ok", result
    return result[1:]


def test_v1_manifest_remains_compatible_for_external_weights(tmp_path):
    weight_path = tmp_path / "external_weights.parquet"
    weights, sell_only = _frames()
    weights.to_parquet(weight_path)
    sell_only.to_parquet(execution.sell_only_path_for_weights(weight_path))

    manifest = execution.write_sell_only_manifest(weight_path)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == execution.LEGACY_SELL_ONLY_MANIFEST_VERSION
    assert "batch_execution_stats_file" not in payload
    assert execution.validate_sell_only_manifest(weight_path) == manifest
    loaded = runmod._load_sell_only_metadata(weight_path)
    pd.testing.assert_frame_equal(loaded, sell_only)


def test_v2_manifest_binds_weights_sell_only_and_execution_stats(tmp_path):
    weight_path = tmp_path / "weights.parquet"

    weights, sell_only, stats, manifest = _publish(weight_path)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == execution.SELL_ONLY_MANIFEST_VERSION
    assert payload["weights_file"] == weights.name
    assert payload["sell_only_file"] == sell_only.name
    assert payload["batch_execution_stats_file"] == stats.name
    assert execution.validate_sell_only_manifest(weight_path) == manifest
    assert not execution.bundle_in_progress_path_for_weights(weight_path).exists()


def test_two_v2_bundles_in_same_directory_keep_independent_stats(tmp_path):
    alpha_path = tmp_path / "alpha_weights.parquet"
    beta_path = tmp_path / "beta_weights.parquet"

    alpha_bundle = _publish(alpha_path)
    beta_bundle = _publish(beta_path, shifted=True)

    assert alpha_bundle[2] == tmp_path / "alpha_weights.batch_execution_stats.json"
    assert beta_bundle[2] == tmp_path / "beta_weights.batch_execution_stats.json"
    assert alpha_bundle[2] != beta_bundle[2]
    assert execution.validate_sell_only_manifest(alpha_path) == alpha_bundle[-1]
    assert execution.validate_sell_only_manifest(beta_path) == beta_bundle[-1]
    assert json.loads(alpha_bundle[2].read_text(encoding="utf-8")) == _stats()
    assert json.loads(beta_bundle[2].read_text(encoding="utf-8")) == _stats(shifted=True)


@pytest.mark.parametrize("damage", ["missing", "tampered"])
def test_v2_manifest_rejects_missing_or_tampered_execution_stats(tmp_path, damage):
    weight_path = tmp_path / "weights.parquet"
    _, _, stats, _ = _publish(weight_path)

    if damage == "missing":
        stats.unlink()
        expected = "批量成交统计缺失"
    else:
        stats.write_text('{"expired_order_count": 999}\n', encoding="utf-8")
        expected = "哈希不匹配"

    with pytest.raises(ValueError, match=expected):
        execution.validate_sell_only_manifest(weight_path)


def test_in_progress_marker_rejects_before_weights_are_parsed(tmp_path, monkeypatch):
    weight_path = tmp_path / "unreadable.parquet"
    execution.mark_bundle_in_progress(weight_path)
    weights_parsed = False

    def fail_if_weights_are_parsed(_path):
        nonlocal weights_parsed
        weights_parsed = True
        raise AssertionError("存在 marker 时不应解析权重")

    monkeypatch.setattr(runmod, "_load_weights", fail_if_weights_are_parsed)

    with pytest.raises(ValueError, match="in-progress"):
        runmod.run_backtest(
            weight_path,
            "2024-01-02",
            "2024-01-31",
            index="all",
        )

    assert not weights_parsed


def test_staging_failure_does_not_change_previously_published_bundle(
    tmp_path,
    monkeypatch,
):
    weight_path = tmp_path / "weights.parquet"
    bundle_paths = _publish(weight_path)
    before = {path: path.read_bytes() for path in bundle_paths}
    original_to_parquet = pd.DataFrame.to_parquet
    new_weights, new_sell_only = _frames(shifted=True)

    def fail_new_weight_staging(self, path, *args, **kwargs):
        if self is new_weights:
            raise OSError("simulated staging failure")
        return original_to_parquet(self, path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_new_weight_staging)

    with pytest.raises(OSError, match="staging failure"):
        execution.publish_batch_bundle(
            new_weights,
            new_sell_only,
            _stats(shifted=True),
            weight_path,
        )

    assert {path: path.read_bytes() for path in bundle_paths} == before
    assert execution.validate_sell_only_manifest(weight_path) == bundle_paths[-1]
    assert not execution.bundle_in_progress_path_for_weights(weight_path).exists()


def test_publish_failure_is_fail_closed_and_next_publish_recovers(
    tmp_path,
    monkeypatch,
):
    weight_path = tmp_path / "weights.parquet"
    _publish(weight_path)
    new_weights, new_sell_only = _frames(shifted=True)
    original_replace = Path.replace

    def fail_official_weight_replace(self, target):
        if Path(target) == weight_path:
            raise OSError("simulated official publish failure")
        return original_replace(self, target)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "replace", fail_official_weight_replace)
        with pytest.raises(OSError, match="official publish failure"):
            execution.publish_batch_bundle(
                new_weights,
                new_sell_only,
                _stats(shifted=True),
                weight_path,
            )

    marker = execution.bundle_in_progress_path_for_weights(weight_path)
    assert marker.exists()
    with pytest.raises(ValueError, match="in-progress"):
        execution.validate_sell_only_manifest(weight_path)
    with pytest.raises(ValueError, match="in-progress"):
        runmod._load_sell_only_metadata(weight_path)

    recovered = execution.publish_batch_bundle(
        new_weights,
        new_sell_only,
        _stats(shifted=True),
        weight_path,
    )

    assert not marker.exists()
    assert execution.validate_sell_only_manifest(weight_path) == recovered[-1]
    pd.testing.assert_frame_equal(pd.read_parquet(weight_path), new_weights)
    assert json.loads(recovered[2].read_text(encoding="utf-8")) == _stats(shifted=True)


def test_shared_reader_lock_prevents_old_sidecar_new_weights_mix(tmp_path):
    weight_path = tmp_path / "weights.parquet"
    _publish(weight_path)
    context = multiprocessing.get_context("spawn")
    reader_ready = context.Event()
    reader_release = context.Event()
    publisher_attempting = context.Event()
    publisher_acquired = context.Event()
    reader_results = context.Queue()
    publisher_results = context.Queue()
    reader = context.Process(
        target=_holding_reader_worker,
        args=(weight_path, reader_ready, reader_release, reader_results),
        name="bundle-reader",
    )
    publisher = context.Process(
        target=_signaled_publisher_worker,
        args=(
            weight_path,
            True,
            publisher_attempting,
            publisher_acquired,
            publisher_results,
        ),
        name="bundle-publisher",
    )

    reader.start()
    assert reader_ready.wait(15)
    publisher.start()
    assert publisher_attempting.wait(15)
    publisher_was_blocked = not publisher_acquired.wait(0.5)
    reader_release.set()

    reader_snapshot = _assert_process_ok(reader, reader_results)
    _assert_process_ok(publisher, publisher_results)

    assert publisher_was_blocked
    assert reader_snapshot == (0.6, False)
    assert execution.validate_sell_only_manifest(weight_path).is_file()
    assert not execution.bundle_in_progress_path_for_weights(weight_path).exists()


def test_concurrent_publishers_serialize_and_readers_never_see_half_bundle(tmp_path):
    weight_path = tmp_path / "weights.parquet"
    _publish(weight_path)
    context = multiprocessing.get_context("spawn")
    first_paused = context.Event()
    first_release = context.Event()
    second_attempting = context.Event()
    second_acquired = context.Event()
    reader_attempting = context.Event()
    reader_acquired = context.Event()
    first_results = context.Queue()
    second_results = context.Queue()
    reader_results = context.Queue()
    first = context.Process(
        target=_mid_publish_worker,
        args=(weight_path, True, first_paused, first_release, first_results),
        name="first-publisher",
    )
    second = context.Process(
        target=_signaled_publisher_worker,
        args=(
            weight_path,
            False,
            second_attempting,
            second_acquired,
            second_results,
        ),
        name="second-publisher",
    )
    reader = context.Process(
        target=_signaled_reader_worker,
        args=(weight_path, reader_attempting, reader_acquired, reader_results),
        name="mid-publish-reader",
    )

    first.start()
    assert first_paused.wait(15)
    assert execution.bundle_in_progress_path_for_weights(weight_path).exists()
    second.start()
    assert second_attempting.wait(15)
    second_was_blocked = not second_acquired.wait(0.5)
    reader.start()
    assert reader_attempting.wait(15)
    reader_was_blocked = not reader_acquired.wait(0.5)
    first_release.set()

    _assert_process_ok(first, first_results)
    _assert_process_ok(second, second_results)
    reader_snapshot = _assert_process_ok(reader, reader_results)

    assert second_was_blocked
    assert reader_was_blocked
    assert reader_snapshot in {(0.5, True), (0.6, False)}
    assert execution.validate_sell_only_manifest(weight_path).is_file()
    assert not execution.bundle_in_progress_path_for_weights(weight_path).exists()
    final_weights, final_sell_only = _frames()
    pd.testing.assert_frame_equal(pd.read_parquet(weight_path), final_weights)
    pd.testing.assert_frame_equal(
        pd.read_parquet(execution.sell_only_path_for_weights(weight_path)),
        final_sell_only,
    )
