"""Terminal logging configuration for HINT application entry points."""

from __future__ import annotations

import logging

_CONFIGURED_ATTR = "_hint_terminal_logging_configured"
_SAM2_NOISE = (
    "For numpy array image",
    "Computing image embeddings",
    "Image embeddings computed",
)


class _Sam2NoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(token in message for token in _SAM2_NOISE)


def configure_project_logging() -> None:
    """Add one INFO terminal handler and suppress verbose dependency logs.

    Call explicitly at application startup. Repeated calls are idempotent;
    existing root handlers are preserved. Messages are written to stderr.
    """
    root = logging.getLogger()
    if getattr(root, _CONFIGURED_ATTR, False):
        return

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.addFilter(_Sam2NoiseFilter())
    handler.setFormatter(
        logging.Formatter("%(message)s"),
    )
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)

    logging.getLogger("sam2").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").disabled = True
    setattr(root, _CONFIGURED_ATTR, True)
