"""FastAPI Backend Server for SMARTCACHE AI.

Provides RESTful endpoints for running cache simulations, comparing replacement
policies against baseline and optimal oracles, and streaming performance metrics.
"""
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from cache_simulator import CacheSimulator, SimulationResult
from policies import (
    BasePolicy,
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
from benchmarks.metrics import compute_extended_metrics, calculate_rolling_hit_rate

# Initialize FastAPI app
app = FastAPI(
    title="SMARTCACHE AI Backend",
    description="RESTful API for simulating and benchmarking adaptive cache replacement policies",
    version="1.0.0",
)

# Enable CORS for frontend and dashboard integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supported policies and workloads lists
AVAILABLE_POLICIES: List[str] = [
    "LRU",
    "LFU",
    "Belady's Algorithm",
    "Smartcache Policy",
    "ARC",
    "SIEVE",
    "Classic LeCaR",
]

AVAILABLE_WORKLOADS: List[str] = [
    "Looping hot set",
    "Sequential scan",
    "Zipf-like skew",
    "Phase shift",
    "Mixed workload",
]


# ============================================================================
# Request / Response Schemas
# ============================================================================

class Simulation(BaseModel):
    """Schema for single simulation execution."""
    workload_type: str = Field(..., description="Type of workload trace to generate")
    cache_capacity: int = Field(..., description="Capacity of the cache (number of items)")
    trace_length: int = Field(..., description="Number of request accesses to simulate")
    policy: str = Field(..., description="Cache replacement policy name")
    random_seed: Optional[int] = Field(0, description="Random seed for reproducible trace and policy generation")
    custom_trace: Optional[List[int]] = Field(None, description="Optional custom trace items list")
    record_events: Optional[bool] = Field(False, description="Whether to record detailed step-by-step events")
    include_events: Optional[bool] = Field(False, description="Whether to return event timeline and rolling hit rate")


class Compare(BaseModel):
    """Schema for multi-policy benchmark comparison."""
    cache_capacity: int = Field(..., description="Capacity of the cache (number of items)")
    workload_type: str = Field(..., description="Type of workload trace to generate")
    trace_length: int = Field(..., description="Number of request accesses to simulate")
    policy: List[str] = Field(..., description="List of policy names to compare")
    random_seed: Optional[int] = Field(0, description="Random seed for reproducible trace generation")
    custom_trace: Optional[List[int]] = Field(None, description="Optional custom trace items list")


# ============================================================================
# Helper Functions
# ============================================================================

def generate_workload_trace(workload_type: str, length: int, seed: int = 42) -> List[int]:
    """Generate deterministic synthetic trace matching requested workload pattern."""
    w_norm = workload_type.strip().lower()

    if "loop" in w_norm:
        hot_set_size = min(10, max(1, length))
        repetitions = max(1, (length // hot_set_size) + 5)
        return generate_looping_trace(
            hot_set_size=hot_set_size,
            repetitions=repetitions,
            noise_ratio=0.05,
            seed=seed,
        )[:length]

    elif "scan" in w_norm:
        return generate_sequential_scan_trace(scan_length=length, seed=seed)[:length]

    elif "zipf" in w_norm:
        return generate_zipf_trace(
            num_items=max(10, length // 5),
            total_requests=length,
            alpha=1.0,
            seed=seed,
        )[:length]

    elif "phase" in w_norm or "shift" in w_norm:
        p1 = max(1, length // 3)
        p2 = max(1, length // 3)
        p3 = max(1, length - (p1 + p2))
        phases = [
            {"type": "loop", "hot_set": list(range(0, 10)), "length": p1},
            {"type": "scan", "start": 1000, "length": p2},
            {"type": "loop", "hot_set": list(range(20, 30)), "length": p3},
        ]
        return generate_phase_shift_trace(phases=phases, seed=seed)[:length]

    elif "mix" in w_norm:
        return generate_mixed_trace(total_requests=length, seed=seed)[:length]

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported workload_type '{workload_type}'. Choose from: {AVAILABLE_WORKLOADS}",
        )


def create_policy_instance(policy_name: str, capacity: int, seed: int = 42) -> BasePolicy:
    """Instantiate a policy instance based on the policy name."""
    p_norm = policy_name.strip().lower()

    if "classic" in p_norm:
        return ClassicLeCaRPolicy(capacity=capacity, seed=seed, name="Classic LeCaR")
    elif "smart" in p_norm or "adaptive" in p_norm or "lecar" in p_norm:
        return LeCaRAdaptivePolicy(capacity=capacity, seed=seed, name="Smartcache Policy")
    elif "lru" in p_norm:
        return LRUPolicy(name="LRU")
    elif "lfu" in p_norm:
        return LFUPolicy(name="LFU")
    elif "belady" in p_norm or "opt" in p_norm:
        return BeladyPolicy(name="Belady's Algorithm")
    elif "arc" in p_norm:
        return ARCPolicy(capacity=capacity, name="ARC")
    elif "sieve" in p_norm:
        return SIEVEPolicy(capacity=capacity, name="SIEVE")
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported policy '{policy_name}'. Choose from: {AVAILABLE_POLICIES}",
        )


def format_simulation_result(
    result: SimulationResult,
    trace: Optional[List[Any]] = None,
    include_events: bool = False,
) -> Dict[str, Any]:
    """Format simulation metrics into a clean dictionary."""
    summary: Dict[str, Any] = {
        "policy_name": result.policy_name,
        "hit_rate": round(result.hit_rate, 4),
        "miss_rate": round(result.miss_rate, 4),
        "total_requests": result.total_requests,
        "cache_hits": result.hits,
        "cache_misses": result.misses,
        "hits": result.hits,
        "misses": result.misses,
        "evictions": result.evictions,
        "wall_clock_time_sec": round(result.wall_clock_time_sec, 6),
        "avg_decision_time_us": round(result.avg_decision_time_us, 2),
    }

    if trace is not None and len(trace) > 0:
        extended = compute_extended_metrics(result, trace)
        summary["compulsory_misses"] = extended.get("compulsory_misses", 0)
        summary["capacity_misses"] = extended.get("capacity_misses", 0)
        summary["compulsory_miss_rate"] = extended.get("compulsory_miss_rate", 0.0)
        summary["capacity_miss_rate"] = extended.get("capacity_miss_rate", 0.0)

    if include_events and result.events:
        summary["events"] = [
            {
                "step": ev.step,
                "item": ev.item,
                "is_hit": ev.is_hit,
                "evicted_item": ev.evicted_item,
                "policy_state": ev.policy_state,
            }
            for ev in result.events
        ]
        summary["rolling_hit_rate"] = [
            round(r, 4) for r in calculate_rolling_hit_rate(result.events)
        ]

    return summary


# ============================================================================
# API Routes
# ============================================================================

@app.get("/")
def read_root():
    """Root status check endpoint."""
    return {"Project": "Smartcache-ai", "status": "Running"}


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Service health check endpoint."""
    return {"status": "Ok"}


@app.get("/policies", status_code=status.HTTP_200_OK)
def get_policies():
    """Retrieve available cache replacement policies."""
    return {"policies": AVAILABLE_POLICIES}


@app.get("/workloads", status_code=status.HTTP_200_OK)
def get_workloads():
    """Retrieve available synthetic workload patterns."""
    return {"workloads": AVAILABLE_WORKLOADS}


@app.post("/simulations", status_code=status.HTTP_200_OK)
def run_simulation(simulation: Simulation):
    """Run a single cache simulation against the specified policy and workload."""
    simulation_dict = simulation.model_dump()
    if simulation.cache_capacity <= 0 or simulation.trace_length <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cache capacity and trace length must be positive integers",
        )

    seed = simulation.random_seed if simulation.random_seed is not None else 0

    # Obtain trace: custom trace or synthetic generator
    if simulation.custom_trace is not None and len(simulation.custom_trace) > 0:
        trace = simulation.custom_trace[: simulation.trace_length]
    else:
        trace = generate_workload_trace(
            workload_type=simulation.workload_type,
            length=simulation.trace_length,
            seed=seed,
        )

    # Instantiate policy
    policy_instance = create_policy_instance(
        policy_name=simulation.policy,
        capacity=simulation.cache_capacity,
        seed=seed,
    )

    # Execute simulation
    record_events = bool(simulation.record_events or simulation.include_events)
    simulator = CacheSimulator(
        capacity=simulation.cache_capacity,
        policy=policy_instance,
        record_events=record_events,
    )
    result = simulator.run(trace)

    formatted_result = format_simulation_result(
        result=result,
        trace=trace,
        include_events=bool(simulation.include_events),
    )

    return {
        "message": "Simulation completed successfully",
        "configuration": simulation_dict,
        "simulation_result": formatted_result,
    }


@app.post("/simulations/compare", status_code=status.HTTP_200_OK)
def compare_simulations(compare_request: Compare):
    """Run fair multi-policy comparison on the identical workload trace."""
    compare_request_dict = compare_request.model_dump()
    if compare_request.cache_capacity <= 0 or compare_request.trace_length <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cache capacity and trace length must be positive integers",
        )

    if not compare_request.policy:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one policy must be specified in the 'policy' list",
        )

    seed = compare_request.random_seed if compare_request.random_seed is not None else 0

    # Obtain trace once for all policies to guarantee fairness
    if compare_request.custom_trace is not None and len(compare_request.custom_trace) > 0:
        trace = compare_request.custom_trace[: compare_request.trace_length]
    else:
        trace = generate_workload_trace(
            workload_type=compare_request.workload_type,
            length=compare_request.trace_length,
            seed=seed,
        )

    # Run simulations across all policies
    comparison_results: Dict[str, Any] = {}
    for policy_name in compare_request.policy:
        policy_instance = create_policy_instance(
            policy_name=policy_name,
            capacity=compare_request.cache_capacity,
            seed=seed,
        )
        simulator = CacheSimulator(
            capacity=compare_request.cache_capacity,
            policy=policy_instance,
            record_events=False,
        )
        result = simulator.run(trace)
        comparison_results[policy_name] = format_simulation_result(
            result=result,
            trace=trace,
            include_events=False,
        )

    return {
        "message": "Comparison successful",
        "configuration": compare_request_dict,
        "comparison_results": comparison_results,
    }