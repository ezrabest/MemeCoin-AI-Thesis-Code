"""AE12.5 Runtime Observability / Final MSc Reporting — read-only layer."""

from __future__ import annotations

from .report_manager import AE12ReportManager, get_ae12_report_manager, reset_ae12_report_manager

__all__ = [
    "AE12ReportManager",
    "get_ae12_report_manager",
    "reset_ae12_report_manager",
]
