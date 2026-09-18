"""Manipulation-pattern prediction, training, annotation, evaluation, and runtime.

The package is intentionally split by responsibility:

``runtime``
    Online manipulation-pattern prediction, stabilization, and routed rendering.
``models`` and ``data``
    Shared model and LeRobot dataset implementations.
``train``, ``annotation``, and ``evaluation``
    Command-line workflows. Train with ``python -m pattern.train``; annotate and
    evaluate with ``python -m pattern.<area>.<command>``.
"""

from __future__ import annotations
