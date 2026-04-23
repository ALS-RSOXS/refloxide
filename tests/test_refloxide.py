"""Tests for refloxide."""

from refloxide import __version__, hello_from_rust


def test_version() -> None:
    """Verify that the package exposes a version string."""
    assert __version__ is not None
    assert isinstance(__version__, str)


def test_rust_extension() -> None:
    """Verify that the Rust extension module can be imported and called."""
    assert hello_from_rust() == "Hello from refloxide core"
