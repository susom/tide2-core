"""
.. include:: ../../README.md
"""

try:
    # Written by hatch-vcs at build time from the git tag (see pyproject.toml).
    from tide2._version import __version__
except ImportError:  # pragma: no cover - source checkout without a build
    __version__ = "0.0.0.dev0"

__author__ = "TiDE-CORE 2.0 Team"
__docformat__ = "google"
