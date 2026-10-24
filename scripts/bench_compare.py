#!/usr/bin/env python3
"""Benchmark the current working tree against a base ref on this machine.

Run this when you open a pull request and paste the printed markdown table
into the PR description. It builds a release wheel for the base ref (in a
temporary git worktree) and for the working tree, runs
``examples/gpu_backend_bench.py`` against each in an isolated environment, and
prints a before/after table with the machine description.

    uv run python scripts/bench_compare.py     # base = merge-base with main
    uv run python scripts/bench_compare.py --base v0.2.1 --repeats 9

Both sides run the benchmark script from the working tree, so the workloads
are identical. Close other heavy programs first; numbers from one machine are
only comparable with each other.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "examples" / "gpu_backend_bench.py"


def run(cmd: list[str], *, cwd: Path = ROOT, quiet: bool = False) -> str:
    """Run ``cmd`` and return stdout, exiting with its output on failure."""
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    if not quiet and proc.stderr.strip():
        print(proc.stderr.strip(), file=sys.stderr)
    return proc.stdout


def resolve_base(requested: str | None) -> str:
    """Commit to benchmark against: ``requested``, else merge-base with main."""
    if requested:
        return run(["git", "rev-parse", "--verify", f"{requested}^{{commit}}"]).strip()
    for ref in ("origin/main", "main"):
        probe = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", ref], cwd=ROOT, check=False
        )
        if probe.returncode == 0:
            return run(["git", "merge-base", "HEAD", ref]).strip()
    sys.exit("no origin/main or main found; pass --base <ref>")


def build_wheel(source: Path, out_dir: Path) -> Path:
    """Release wheel for the project at ``source``."""
    print(f"building release wheel for {source} ...", flush=True)
    run(["uvx", "maturin", "build", "--release", "-o", str(out_dir)], cwd=source)
    wheels = sorted(out_dir.glob("*.whl"))
    if not wheels:
        sys.exit(f"no wheel produced in {out_dir}")
    return wheels[-1]


def bench_wheel(wheel: Path, out_json: Path, assets: Path, args: argparse.Namespace):
    """Run the benchmark script against ``wheel`` in an isolated environment."""
    print(f"benchmarking {wheel.name} ...", flush=True)
    run(
        [
            "uv", "run", "--isolated", "--no-project",
            "--with", "numpy", "--with", str(wheel),
            "python", str(BENCH),
            "--repeats", str(args.repeats),
            "--warmup", str(args.warmup),
            "--assets-dir", str(assets),
            "--benchmark-json", str(out_json),
        ],
        quiet=True,
    )  # fmt: skip
    return {e["name"]: e for e in json.loads(out_json.read_text(encoding="utf-8"))}


def machine_description() -> list[str]:
    """Human-readable machine facts for the PR table header."""
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        cpu = run(["sysctl", "-n", "machdep.cpu.brand_string"], quiet=True).strip()
    elif sys.platform.startswith("linux"):
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    rustc = run(["rustc", "--version"], quiet=True).strip()
    return [
        f"machine: {platform.system()} {platform.machine()}, {cpu}, "
        f"{os.cpu_count()} logical cores",
        f"toolchain: {rustc}",
        f"date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
    ]


def format_table(base: dict, head: dict) -> list[str]:
    """Markdown rows comparing every benchmark present on both sides."""
    lines = [
        "| benchmark | base | head | change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in head:
        if name not in base:
            continue
        before, after = base[name]["value"], head[name]["value"]
        unit = head[name]["unit"]
        change = (after / before - 1.0) * 100.0 if before else float("nan")
        lines.append(
            f"| {name} | {before:.3f} {unit} | {after:.3f} {unit} | {change:+.1f}% |"
        )
    return lines


def main() -> None:
    """Build both sides, benchmark them, and print the markdown report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", help="base ref (default: merge-base with main)")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "bench-compare.md", help="markdown report"
    )
    args = parser.parse_args()

    base_sha = resolve_base(args.base)
    head_sha = run(["git", "rev-parse", "HEAD"]).strip()
    status = run(["git", "status", "--porcelain", "--untracked-files=no"])
    dirty = bool(status.strip())
    suffix = " + uncommitted" if dirty else ""
    print(f"base {base_sha[:9]}  head {head_sha[:9]}{suffix}")

    tmp = Path(tempfile.mkdtemp(prefix="refloxide-bench-"))
    worktree = tmp / "base"
    try:
        run(["git", "worktree", "add", "--detach", str(worktree), base_sha])
        base_wheel = build_wheel(worktree, tmp / "wheel-base")
        head_wheel = build_wheel(ROOT, tmp / "wheel-head")
        base = bench_wheel(base_wheel, tmp / "base.json", tmp / "assets-base", args)
        head = bench_wheel(head_wheel, tmp / "head.json", tmp / "assets-head", args)
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        shutil.rmtree(tmp, ignore_errors=True)

    report = [
        "### Benchmark (before / after, one machine)",
        "",
        f"base `{base_sha[:9]}`, head `{head_sha[:9]}`"
        f"{' plus uncommitted changes' if dirty else ''}; "
        f"median of {args.repeats} runs after {args.warmup} warmup",
        "",
        *[f"- {line}" for line in machine_description()],
        "",
        *format_table(base, head),
        "",
    ]
    text = "\n".join(report)
    args.out.write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
