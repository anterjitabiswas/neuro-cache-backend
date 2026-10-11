"""Workload generator and loader utilities for SMARTCACHE AI."""
from workloads.generators import (
    generate_looping_trace,
    generate_sequential_scan_trace,
    generate_zipf_trace,
    generate_phase_shift_trace,
    generate_mixed_trace,
)

__all__ = [
    "generate_looping_trace",
    "generate_sequential_scan_trace",
    "generate_zipf_trace",
    "generate_phase_shift_trace",
    "generate_mixed_trace",
]
