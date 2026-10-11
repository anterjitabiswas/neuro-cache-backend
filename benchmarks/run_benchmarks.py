"""Reproducible benchmark runner for multi-policy evaluation across modern and classic competitors."""
import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from cache_simulator import CacheSimulator
from policies import (
    LRUPolicy,
    LFUPolicy,
    BeladyPolicy,
    LeCaRAdaptivePolicy,
    ARCPolicy,
    SIEVEPolicy,
    ClassicLeCaRPolicy,
)
from workloads.generators import (
    generate_looping_trace,
    generate_sequential_scan_trace,
    generate_zipf_trace,
    generate_phase_shift_trace,
    generate_mixed_trace,
)
from benchmarks.metrics import (
    compute_extended_metrics,
    calculate_adaptation_penalty,
    estimate_metadata_overhead,
)


def run_benchmark_suite(output_dir: str = "results", seed: int = 42) -> pd.DataFrame:
    """Run all 7 policies across standard workloads and save structured results."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Define standard workloads
    workloads: Dict[str, List[int]] = {
        "Phase Shift (Shift)": generate_phase_shift_trace(seed=seed),
        "Looping (Hot Set)": generate_looping_trace(hot_set_size=10, repetitions=50, noise_ratio=0.05, seed=seed),
        "Zipf Skew (alpha=1.0)": generate_zipf_trace(num_items=100, total_requests=1000, alpha=1.0, seed=seed),
        "Mixed Workload": generate_mixed_trace(total_requests=1000, seed=seed),
        "Sequential Scan": generate_sequential_scan_trace(scan_length=500, seed=seed),
    }

    # 2. Test Capacities
    capacities = [5, 10, 20]
    records: List[Dict[str, Any]] = []

    print("=" * 95)
    print("SMARTCACHE AI — EXTENDED MULTI-POLICY BENCHMARK SUITE (MODERN & CLASSIC COMPETITORS)")
    print("=" * 95)

    for workload_name, trace in workloads.items():
        print(f"\n--- Workload: {workload_name} (Length: {len(trace)}) ---")

        for cap in capacities:
            # Instantiate all 7 policies
            policies = [
                LRUPolicy(),
                LFUPolicy(),
                SIEVEPolicy(capacity=cap),
                ARCPolicy(capacity=cap),
                ClassicLeCaRPolicy(capacity=cap, seed=seed),
                LeCaRAdaptivePolicy(capacity=cap, seed=seed, name="SmartCache AI"),
                BeladyPolicy(trace=trace),
            ]

            for policy in policies:
                # Need events for adaptation penalty measurement on Phase Shift
                record_events = (workload_name == "Phase Shift (Shift)")
                sim = CacheSimulator(capacity=cap, policy=policy, record_events=record_events)
                result = sim.run(trace)
                extended = compute_extended_metrics(result, trace)

                # Adaptation penalty calculation (for Phase Shift)
                adapt_penalty = 0
                if workload_name == "Phase Shift (Shift)":
                    adapt_penalty = calculate_adaptation_penalty(result.events, shift_start_idx=600, window_length=100, min_compulsory=10)

                metadata_entries = estimate_metadata_overhead(policy, cap)

                record = {
                    "Workload": workload_name,
                    "Capacity": cap,
                    "Policy": policy.name,
                    "Requests": result.total_requests,
                    "Hits": result.hits,
                    "Misses": result.misses,
                    "Hit Rate (%)": round(result.hit_rate * 100, 2),
                    "Miss Rate (%)": round(result.miss_rate * 100, 2),
                    "Evictions": result.evictions,
                    "Capacity Misses": extended["capacity_misses"],
                    "Metadata Entries": metadata_entries,
                    "Adapt Penalty Misses": adapt_penalty if workload_name == "Phase Shift (Shift)" else None,
                    "Avg Decision (us)": result.avg_decision_time_us,
                    "Wall Clock (s)": round(result.wall_clock_time_sec, 4),
                }
                records.append(record)

    df = pd.DataFrame(records)

    # Save outputs
    csv_file = out_path / "benchmark_summary.csv"
    json_file = out_path / "benchmark_summary.json"
    df.to_csv(csv_file, index=False)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print("\nBenchmark Suite Completed successfully!")
    print(f"Results saved to:\n  - {csv_file}\n  - {json_file}")

    # Print formatted pivot table for Hit Rate
    pivot_hr = df.pivot_table(
        index=["Workload", "Capacity"],
        columns="Policy",
        values="Hit Rate (%)",
    )
    print("\n--- Summary Hit Rate (%) Across All Competitors ---")
    print(pivot_hr.to_string())

    # Print Adaptation Penalty on Phase Shift
    phase_shift_df = df[df["Workload"] == "Phase Shift (Shift)"]
    pivot_adapt = phase_shift_df.pivot_table(
        index=["Capacity"],
        columns="Policy",
        values="Adapt Penalty Misses",
    )
    print("\n--- Phase Shift Adaptation Penalty (Extra Misses during Shift Recovery) ---")
    print(pivot_adapt.to_string())

    return df


if __name__ == "__main__":
    run_benchmark_suite()
