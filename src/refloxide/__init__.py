"""Public Python interface for refloxide."""

from ._core import hello_from_rust

__all__ = ["__version__", "hello_from_rust"]
__version__ = "0.1.0"
