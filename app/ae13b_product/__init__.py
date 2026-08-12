"""AE13B product package — demo bot, guards, presets, audit runner."""

from __future__ import annotations

from app.ae13b_product.execution_guard import (
    DemoExecutionGuardError,
    assert_paper_demo_allowed,
    evaluate_paper_demo_execution_guard,
)

__all__ = [
    "DemoExecutionGuardError",
    "assert_paper_demo_allowed",
    "evaluate_paper_demo_execution_guard",
]
