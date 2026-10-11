"""Deterministic synthetic workload trace generators for SMARTCACHE AI."""
import random
from typing import Any, Dict, List, Optional
import numpy as np


def generate_looping_trace(
    hot_set_size: int = 10,
    repetitions: int = 50,
    noise_ratio: float = 0.05,
    noise_universe: int = 500,
    seed: int = 42,
) -> List[int]:
    """Generate a looping trace with recurring hot items and optional noise."""
    rng = random.Random(seed)
    hot_set = list(range(hot_set_size))
    trace: List[int] = []

    for _ in range(repetitions):
        for item in hot_set:
            if rng.random() < noise_ratio:
                # Inject cold random noise item
                noise_item = 10000 + rng.randint(0, noise_universe)
                trace.append(noise_item)
            else:
                trace.append(item)

    return trace


def generate_sequential_scan_trace(
    scan_length: int = 500,
    repeat_cycle: int = 0,
    seed: int = 42,
) -> List[int]:
    """Generate a sequential scan trace through unique item IDs."""
    if repeat_cycle > 0:
        cycles = max(1, scan_length // repeat_cycle)
        trace = []
        for _ in range(cycles):
            trace.extend(list(range(repeat_cycle)))
        return trace
    return list(range(scan_length))


def generate_zipf_trace(
    num_items: int = 100,
    total_requests: int = 1000,
    alpha: float = 1.0,
    seed: int = 42,
) -> List[int]:
    """Generate a Zipf-distributed popularity trace (power law)."""
    np_rng = np.random.default_rng(seed)
    # Calculate Zipf probabilities P(k) ~ 1 / k^alpha
    ranks = np.arange(1, num_items + 1)
    weights = 1.0 / (ranks ** alpha)
    probabilities = weights / np.sum(weights)

    # Sample item indices (0 to num_items - 1)
    samples = np_rng.choice(np.arange(num_items), size=total_requests, p=probabilities)
    return samples.tolist()


def generate_phase_shift_trace(
    phases: Optional[List[Dict[str, Any]]] = None,
    seed: int = 42,
) -> List[int]:
    """Generate a workload exhibiting sudden phase transitions.

    Default phases:
      Phase 1: Hot Set A (items 0..9) for 300 requests
      Phase 2: Sequential Scan (items 1000..1300) for 300 requests
      Phase 3: Hot Set B (items 20..29) for 300 requests
    """
    if phases is None:
        phases = [
            {"type": "loop", "hot_set": list(range(0, 10)), "length": 300},
            {"type": "scan", "start": 1000, "length": 300},
            {"type": "loop", "hot_set": list(range(20, 30)), "length": 300},
        ]

    rng = random.Random(seed)
    trace: List[int] = []

    for phase in phases:
        ptype = phase.get("type", "loop")
        length = phase.get("length", 300)

        if ptype == "loop":
            hot_set = phase.get("hot_set", list(range(10)))
            for _ in range(length):
                trace.append(rng.choice(hot_set))
        elif ptype == "scan":
            start = phase.get("start", 1000)
            trace.extend(list(range(start, start + length)))
        elif ptype == "zipf":
            num_items = phase.get("num_items", 50)
            alpha = phase.get("alpha", 1.0)
            sub_trace = generate_zipf_trace(num_items=num_items, total_requests=length, alpha=alpha, seed=rng.randint(0, 10000))
            trace.extend(sub_trace)

    return trace


def generate_mixed_trace(
    total_requests: int = 1000,
    seed: int = 42,
) -> List[int]:
    """Generate a mixed trace with periodic loops, zipf popularity, and scan bursts."""
    rng = random.Random(seed)
    hot_set = list(range(15))
    scan_counter = 5000
    trace: List[int] = []

    # Pre-generate zipf distribution for background requests
    zipf_pool = generate_zipf_trace(num_items=100, total_requests=total_requests, alpha=1.1, seed=seed)
    zipf_idx = 0

    while len(trace) < total_requests:
        dice = rng.random()
        if dice < 0.50:
            # Hot set access (50% probability)
            trace.append(rng.choice(hot_set))
        elif dice < 0.85:
            # Zipf background access (35% probability)
            trace.append(zipf_pool[zipf_idx % len(zipf_pool)])
            zipf_idx += 1
        else:
            # Sequential scan burst (15% probability, 5 items)
            for _ in range(5):
                if len(trace) >= total_requests:
                    break
                trace.append(scan_counter)
                scan_counter += 1

    return trace[:total_requests]
