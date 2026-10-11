"""Evaluation metrics calculation for SMARTCACHE AI."""
from typing import Any, Dict, List, Set
import numpy as np

from cache_simulator import SimulationResult, SimulationEvent


def compute_extended_metrics(result: SimulationResult, trace: List[Any]) -> Dict[str, Any]:
    """Compute detailed evaluation metrics including compulsory vs capacity misses."""
    seen_items: Set[Any] = set()
    compulsory_misses = 0

    for item in trace:
        if item not in seen_items:
            compulsory_misses += 1
            seen_items.add(item)

    capacity_misses = max(0, result.misses - compulsory_misses)

    summary = result.to_dict()
    summary["compulsory_misses"] = compulsory_misses
    summary["capacity_misses"] = capacity_misses
    summary["compulsory_miss_rate"] = round(compulsory_misses / result.total_requests, 4) if result.total_requests > 0 else 0.0
    summary["capacity_miss_rate"] = round(capacity_misses / result.total_requests, 4) if result.total_requests > 0 else 0.0

    return summary


def calculate_rolling_hit_rate(events: List[SimulationEvent], window_size: int = 50) -> List[float]:
    """Calculate rolling window hit rate across the timeline of events."""
    if not events:
        return []

    hits = np.array([1.0 if ev.is_hit else 0.0 for ev in events])
    if len(hits) < window_size:
        return np.cumsum(hits) / np.arange(1, len(hits) + 1)

    kernel = np.ones(window_size) / window_size
    rolling = np.convolve(hits, kernel, mode="valid")
    # Prefix with expanding average up to window_size - 1
    prefix = np.cumsum(hits[: window_size - 1]) / np.arange(1, window_size)
    return np.concatenate([prefix, rolling]).tolist()


def calculate_adaptation_penalty(
    events: List[SimulationEvent],
    shift_start_idx: int = 600,
    window_length: int = 100,
    min_compulsory: int = 10,
) -> int:
    """Calculate the number of extra misses suffered during the shift recovery window."""
    if not events or len(events) < shift_start_idx:
        return 0

    sub_events = events[shift_start_idx : shift_start_idx + window_length]
    misses = sum(1 for ev in sub_events if not ev.is_hit)
    return max(0, misses - min_compulsory)


def estimate_metadata_overhead(policy: Any, capacity: int) -> int:
    """Estimate total metadata entries (cache items + ghost entries + trackers) tracked by policy."""
    state = policy.get_state() if hasattr(policy, "get_state") else {}
    if "total_metadata_entries" in state:
        return int(state["total_metadata_entries"])

    # Fallbacks based on policy attributes
    if hasattr(policy, "h_lru") and hasattr(policy, "h_lfu"):
        entries = capacity + len(policy.h_lru) + len(policy.h_lfu)
        if hasattr(policy, "recent_outcomes"):
            entries += len(policy.recent_outcomes)
        return entries
    elif hasattr(policy, "t1") and hasattr(policy, "b1"):
        return len(policy.t1) + len(policy.b1) + len(policy.t2) + len(policy.b2)
    elif hasattr(policy, "nodes"):
        return len(policy.nodes)
    elif hasattr(policy, "freq_map"):
        return len(policy.freq_map)
    elif hasattr(policy, "order"):
        return len(policy.order)
    return capacity
