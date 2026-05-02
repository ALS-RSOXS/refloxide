"""MkDocs hooks: materialize rustdoc under docs/ so links validate and publish."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger("mkdocs")


def _write_rustdoc_stub(dest: Path) -> None:
    stub = dest / "_core" / "index.html"
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_text(
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        "<title>Rustdoc not generated</title></head><body>"
        "<p>Rust documentation was not copied. Install a Rust toolchain and run "
        "<code>mkdocs build</code> again, or run <code>cargo doc --no-deps</code> "
        "and open <code>target/doc/_core/index.html</code>.</p>"
        "</body></html>\n",
        encoding="utf-8",
    )


def on_pre_build(config, **_kwargs) -> None:
    cfg_path = Path(config.config_file_path or "mkdocs.yml").resolve()
    root = cfg_path.parent
    docs_dir = Path(config.docs_dir).resolve()
    dest = docs_dir / "rustdoc"
    cargo_toml = root / "Cargo.toml"
    target_doc = root / "target" / "doc"
    if dest.exists():
        shutil.rmtree(dest)
    if not cargo_toml.is_file():
        logger.info("rustdoc: no Cargo.toml at %s; stub only", root)
        _write_rustdoc_stub(dest)
        return
    try:
        subprocess.run(
            ["cargo", "doc", "--no-deps"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning(
            "rustdoc: cargo doc unavailable or failed (%s); using stub page",
            exc,
        )
        _write_rustdoc_stub(dest)
        return
    if not target_doc.is_dir():
        logger.warning("rustdoc: %s missing after cargo doc; using stub", target_doc)
        _write_rustdoc_stub(dest)
        return
    shutil.copytree(target_doc, dest)
    logger.info("rustdoc: materialized under %s", dest)
