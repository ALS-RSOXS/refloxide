"""Regression and ranking gates for the uniaxial backend bench."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_BENCH = Path(__file__).resolve().parents[1] / "examples" / "gpu_backend_bench.py"
_SPEC = importlib.util.spec_from_file_location("gpu_backend_bench", _BENCH)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)

BACKEND_GPU = _MOD.BACKEND_GPU
BACKEND_PARALLEL = _MOD.BACKEND_PARALLEL
BACKEND_PYPXR = _MOD.BACKEND_PYPXR
BACKEND_SERIAL = _MOD.BACKEND_SERIAL
BenchRow = _MOD.BenchRow
baseline_regressions = _MOD.baseline_regressions
benchmark_action_entries = _MOD.benchmark_action_entries
evaluate_gates = _MOD.evaluate_gates
ranking_failures = _MOD.ranking_failures
write_benchmark_action_json = _MOD.write_benchmark_action_json


def _row(backend: str, median_s: float, *, rss_mib: float = 40.0) -> BenchRow:
    return BenchRow(
        backend=backend,
        n_q=256,
        n_film=10_000,
        n_layers=10_002,
        median_s=median_s,
        peak_heap_mib=0.0,
        rss_mib=rss_mib,
    )


def test_ranking_passes_when_gpu_wins() -> None:
    rows = [
        _row(BACKEND_GPU, 0.004),
        _row(BACKEND_PARALLEL, 0.14),
        _row(BACKEND_SERIAL, 0.78),
        _row(BACKEND_PYPXR, 8.6),
    ]
    assert ranking_failures(rows) == []


def test_ranking_fails_when_order_inverts() -> None:
    rows = [
        _row(BACKEND_GPU, 1.0),
        _row(BACKEND_PARALLEL, 0.14),
        _row(BACKEND_SERIAL, 0.78),
        _row(BACKEND_PYPXR, 8.6),
    ]
    failures = ranking_failures(rows)
    assert any("ranking" in item for item in failures)


def test_baseline_regression_detects_slowdown() -> None:
    baseline = [_row(BACKEND_GPU, 0.004), _row(BACKEND_PARALLEL, 0.14)]
    current = [_row(BACKEND_GPU, 0.010), _row(BACKEND_PARALLEL, 0.14)]
    failures = baseline_regressions(current, baseline, max_time_ratio=1.35)
    assert len(failures) == 1
    assert "GPU" in failures[0]


def test_baseline_allows_improvements() -> None:
    baseline = [_row(BACKEND_GPU, 0.004, rss_mib=50.0)]
    current = [_row(BACKEND_GPU, 0.003, rss_mib=40.0)]
    assert baseline_regressions(current, baseline) == []


def test_evaluate_gates_combines_checks() -> None:
    baseline = [
        _row(BACKEND_GPU, 0.004),
        _row(BACKEND_PARALLEL, 0.14),
        _row(BACKEND_SERIAL, 0.78),
        _row(BACKEND_PYPXR, 8.6),
    ]
    current = list(baseline)
    assert (
        evaluate_gates(
            current,
            baseline=baseline,
            enforce_ranking=True,
            min_gpu_speedup=50.0,
            max_time_ratio=1.35,
            max_rss_ratio=2.0,
        )
        == []
    )


def test_benchmark_action_json_is_custom_smaller_is_better(tmp_path) -> None:
    rows = [
        _row(BACKEND_GPU, 0.004, rss_mib=44.0),
        _row(BACKEND_PARALLEL, 0.14, rss_mib=32.0),
        _row(BACKEND_SERIAL, 0.78, rss_mib=32.0),
        _row(BACKEND_PYPXR, 8.6, rss_mib=2000.0),
    ]
    entries = benchmark_action_entries(rows)
    names = {e["name"] for e in entries}
    assert "10k / GPU / time" in names
    assert "10k / GPU / rss" in names
    assert "10k / PyPXR / time" in names
    gpu_time = next(e for e in entries if e["name"] == "10k / GPU / time")
    assert gpu_time["unit"] == "ms"
    assert abs(float(gpu_time["value"]) - 4.0) < 1e-9
    path = write_benchmark_action_json(rows, tmp_path / "benchmark-action.json")
    assert path.is_file()
    assert path.read_text(encoding="utf-8").lstrip().startswith("[")
