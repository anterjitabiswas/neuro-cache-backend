"""Unit tests for LeCaRAdaptivePolicy."""
import math
import pytest
from cache_simulator import CacheSimulator
from policies.adaptive import LeCaRAdaptivePolicy
from workloads.generators import generate_looping_trace, generate_phase_shift_trace


def test_initial_weights():
    """Weights must initialize to equal 0.5."""
    policy = LeCaRAdaptivePolicy(capacity=10)
    assert math.isclose(policy.w_lru, 0.5, rel_tol=1e-5)
    assert math.isclose(policy.w_lfu, 0.5, rel_tol=1e-5)


def test_weights_normalization_and_invariants():
    """Weights must remain finite, normalized, and strictly sum to 1.0."""
    policy = LeCaRAdaptivePolicy(capacity=5, seed=123)
    sim = CacheSimulator(capacity=5, policy=policy)
    trace = [1, 2, 3, 4, 5, 6, 1, 2, 7, 8, 3, 4, 9, 10, 1, 2]
    sim.run(trace)

    for _, w_lru, w_lfu in policy.weight_history:
        assert not math.isnan(w_lru)
        assert not math.isnan(w_lfu)
        assert w_lru >= 0.0
        assert w_lfu >= 0.0
        assert math.isclose(w_lru + w_lfu, 1.0, rel_tol=1e-4)


def test_ghost_cache_bounded_capacity():
    """Ghost caches must never exceed their configured capacity."""
    capacity = 4
    multiplier = 1.5
    policy = LeCaRAdaptivePolicy(capacity=capacity, history_multiplier=multiplier)
    sim = CacheSimulator(capacity=capacity, policy=policy)

    # Long sequential scan causes many evictions
    trace = list(range(100))
    sim.run(trace)

    max_ghost_cap = max(1, int(capacity * multiplier))
    assert len(policy.h_lru) <= max_ghost_cap
    assert len(policy.h_lfu) <= max_ghost_cap


def test_reproducibility_with_seed():
    """Same seed must produce exactly identical results and weight history."""
    trace = generate_looping_trace(hot_set_size=10, repetitions=20, noise_ratio=0.1, seed=42)

    p1 = LeCaRAdaptivePolicy(capacity=5, seed=999)
    res1 = CacheSimulator(capacity=5, policy=p1).run(trace)

    p2 = LeCaRAdaptivePolicy(capacity=5, seed=999)
    res2 = CacheSimulator(capacity=5, policy=p2).run(trace)

    assert res1.hits == res2.hits
    assert res1.misses == res2.misses
    assert res1.evictions == res2.evictions
    assert [w[1] for w in p1.weight_history] == [w[1] for w in p2.weight_history]


def test_single_miss_damping_prevents_knee_jerk_flip():
    """An isolated single ghost miss on dominant expert should apply dampened update."""
    policy = LeCaRAdaptivePolicy(capacity=5, learning_rate=0.4, damping_factor=0.25)
    # Artificially set dominant weight for LFU
    policy.w_lfu = 0.80
    policy.w_lru = 0.20
    policy.baseline_hit_rate = 0.85
    policy.current_step = 20

    # Add isolated item to LFU ghost
    policy.h_lfu["noise_item"] = 18

    # Process miss on noise_item
    policy.on_miss("noise_item", evicted_item=None)

    # LFU should remain dominant (not flipped by an isolated noise miss)
    assert policy.w_lfu > 0.65
    assert policy.w_lfu > policy.w_lru


def test_workload_shift_detection():
    """Sudden scan burst or hit rate collapse activates is_shift_active."""
    policy = LeCaRAdaptivePolicy(capacity=5, shift_threshold=0.20)
    sim = CacheSimulator(capacity=5, policy=policy)

    # First, warm up with a stable hot set to establish baseline
    warm_up = [1, 2, 3, 1, 2, 3, 1, 2, 3] * 5
    sim.run(warm_up)

    # Now feed unique scan items (causing 100% misses)
    for scan_item in range(500, 530):
        sim.step(scan_item)

    # Workload shift detector must have flagged the change
    state = policy.get_state()
    assert state["is_shift_active"] is True


def test_measurable_policy_switching():
    """Policy switching must record adaptation latency and extra misses."""
    policy = LeCaRAdaptivePolicy(capacity=5, seed=42)
    sim = CacheSimulator(capacity=5, policy=policy)
    trace = generate_phase_shift_trace(seed=42)
    sim.run(trace)

    state = policy.get_state()
    assert state["completed_transitions"] > 0
    assert state["avg_adaptation_latency"] > 0

    first_trans = policy.completed_transitions[0]
    assert first_trans["start_step"] >= 0
    assert first_trans["end_step"] > first_trans["start_step"]
    assert first_trans["duration_requests"] > 0
    assert first_trans["extra_misses"] >= 0
    assert first_trans["stabilized"] is True


def test_ghost_cache_hit_purges_entry():
    """A cache hit on an item must immediately purge it from ghost caches."""
    policy = LeCaRAdaptivePolicy(capacity=5)
    policy.h_lru["itemX"] = 5
    policy.h_lfu["itemX"] = 5

    policy.on_hit("itemX")

    assert "itemX" not in policy.h_lru
    assert "itemX" not in policy.h_lfu


def test_ghost_cache_expired_entries_ignored():
    """Ghost entries older than ghost_ttl must expire without penalizing the expert."""
    policy = LeCaRAdaptivePolicy(capacity=5, ttl_multiplier=2.0)
    # ghost_ttl = max(10, 5 * 2) = 10
    policy.h_lru["old_item"] = 10
    policy.current_step = 100  # Delta t = 90 >> ghost_ttl (10)

    initial_w_lru = policy.w_lru

    # Miss on old_item
    policy.on_miss("old_item", evicted_item=None)

    # LRU should NOT be penalized
    assert policy.expired_ghost_entries == 1
    assert math.isclose(policy.w_lru, initial_w_lru, rel_tol=1e-5)


def test_consensus_eviction_neutrality():
    """Consensus evictions (both experts agree) must not create biased ghost entries."""
    policy = LeCaRAdaptivePolicy(capacity=2)
    sim = CacheSimulator(capacity=2, policy=policy)

    # Trace [1, 2, 3]: at request 3, cache is full {1, 2}.
    # Both LRU and LFU agree that item 1 is the victim (least recent and freq 1).
    sim.run([1, 2, 3])

    # Consensus victim (item 1) should not be inserted into either biased ghost cache
    assert "1" not in policy.h_lru
    assert "1" not in policy.h_lfu
