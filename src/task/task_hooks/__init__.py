"""Per-task Python that YAML cannot express.

Each file matches a ``configs/tasks/<name>.yaml`` entry and exports ``HOOKS``.
Sorting tasks have no module here. Keep this package free of the reset/spec
pipeline in ``src/task/*.py``.
"""
