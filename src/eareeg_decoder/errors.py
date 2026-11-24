"""Exceptions with stable, greppable status prefixes."""
from __future__ import annotations


class DataNotProvidedError(RuntimeError):
    """Raised when no authorized data source was supplied. Message starts with ``DATA_NOT_PROVIDED``."""

    def __init__(self, detail: str):
        super().__init__(f"DATA_NOT_PROVIDED: {detail}")


class MissingDependencyError(ImportError):
    """Raised when an optional dependency is absent. Message starts with ``BLOCKED_MISSING_DEPENDENCY``.

    The package never installs anything; the message tells the reader which optional extra to install.
    """

    def __init__(self, package: str, purpose: str, extra: str):
        super().__init__(
            f"BLOCKED_MISSING_DEPENDENCY: '{package}' is required for {purpose}. "
            f"Install it yourself (for example: pip install '{package}' or the '{extra}' extra); "
            "this package does not install dependencies."
        )
        self.package = package


class SchemaError(ValueError):
    """Raised when user-supplied arrays do not satisfy the data contract (see docs/DATA_ACCESS.md)."""


class LeakageError(AssertionError):
    """Raised when a train/test split shares participants, stimulus groups or stimulus keys it must not share."""
