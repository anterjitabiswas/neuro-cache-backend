"""Cache Simulator Engine for SMARTCACHE AI.

Manages cache membership, request processing, hit/miss accounting,
and event logging, keeping policy logic completely decoupled.
"""
from dataclasses import dataclass, field
import time
from typing import Any, Dict, Iterable, List, Optional, Set

from policies.base import BasePolicy


@dataclass
class SimulationEvent:
    """Represents a single request processing event."""
    step: int
    item: Any
    is_hit: bool
    evicted_item: Optional[Any]
    policy_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Encapsulates the aggregated performance metrics and telemetry of a simulation run."""
    policy_name: str
    capacity: int
    total_requests: int
    hits: int
    misses: int
    hit_rate: float
    miss_rate: float
    evictions: int
    wall_clock_time_sec: float
    avg_decision_time_us: float
    events: List[SimulationEvent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a summary dictionary."""
        return {
            "policy_name": self.policy_name,
            "capacity": self.capacity,
            "total_requests": self.total_requests,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "miss_rate": round(self.miss_rate, 4),
            "evictions": self.evictions,
            "wall_clock_time_sec": round(self.wall_clock_time_sec, 6),
            "avg_decision_time_us": round(self.avg_decision_time_us, 2),
        }


class CacheSimulator:
    """Decoupled Cache Simulator Engine."""

    def __init__(self, capacity: int, policy: BasePolicy, record_events: bool = True) -> None:
        if capacity < 1:
            raise ValueError(f"Cache capacity must be at least 1, got {capacity}")
        self.capacity = capacity
        self.policy = policy
        self.record_events = record_events

        # Simulator-owned state
        self.cache: Set[Any] = set()
        self.total_requests: int = 0
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.total_decision_time_ns: int = 0
        self.events: List[SimulationEvent] = []

    def reset(self) -> None:
        """Reset simulator and policy states."""
        self.cache.clear()
        self.total_requests = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_decision_time_ns = 0
        self.events.clear()
        self.policy.reset()

    def step(self, item: Any, step_idx: Optional[int] = None) -> SimulationEvent:
        """Process a single item request through the cache lifecycle."""
        if step_idx is None:
            step_idx = self.total_requests

        self.total_requests += 1
        context = {
            "step": step_idx,
            "capacity": self.capacity,
            "current_cache_size": len(self.cache),
            "requested_item": item,
        }

        if item in self.cache:
            # Cache Hit
            self.hits += 1
            t0 = time.perf_counter_ns()
            self.policy.on_hit(item, context)
            dt = time.perf_counter_ns() - t0
            self.total_decision_time_ns += dt

            event = SimulationEvent(
                step=step_idx,
                item=item,
                is_hit=True,
                evicted_item=None,
                policy_state=self.policy.get_state(),
            )
        else:
            # Cache Miss
            self.misses += 1
            evicted_item = None

            if len(self.cache) >= self.capacity:
                # Eviction required
                self.evictions += 1
                t0 = time.perf_counter_ns()
                victim = self.policy.choose_victim(set(self.cache), context)
                dt = time.perf_counter_ns() - t0
                self.total_decision_time_ns += dt

                if victim in self.cache:
                    self.cache.remove(victim)
                else:
                    # Fallback in case victim wasn't in cache
                    victim = next(iter(self.cache))
                    self.cache.remove(victim)
                evicted_item = victim

            self.cache.add(item)
            t0 = time.perf_counter_ns()
            self.policy.on_miss(item, evicted_item, context)
            dt = time.perf_counter_ns() - t0
            self.total_decision_time_ns += dt

            event = SimulationEvent(
                step=step_idx,
                item=item,
                is_hit=False,
                evicted_item=evicted_item,
                policy_state=self.policy.get_state(),
            )

        if self.record_events:
            self.events.append(event)

        return event

    def run(self, trace: Iterable[Any]) -> SimulationResult:
        """Execute a full trace through the simulator."""
        self.reset()
        trace_list = list(trace)

        # Notify policy of full trace if supported (e.g. Belady)
        if hasattr(self.policy, "set_trace"):
            self.policy.set_trace(trace_list)

        t_start = time.perf_counter()
        for idx, item in enumerate(trace_list):
            self.step(item, step_idx=idx)
        total_wall_clock = time.perf_counter() - t_start

        hit_rate = (self.hits / self.total_requests) if self.total_requests > 0 else 0.0
        miss_rate = (self.misses / self.total_requests) if self.total_requests > 0 else 0.0
        avg_decision_us = (
            (self.total_decision_time_ns / self.total_requests) / 1000.0
            if self.total_requests > 0
            else 0.0
        )

        return SimulationResult(
            policy_name=self.policy.name,
            capacity=self.capacity,
            total_requests=self.total_requests,
            hits=self.hits,
            misses=self.misses,
            hit_rate=hit_rate,
            miss_rate=miss_rate,
            evictions=self.evictions,
            wall_clock_time_sec=total_wall_clock,
            avg_decision_time_us=avg_decision_us,
            events=list(self.events),
        )
