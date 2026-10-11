"""Unit tests for CacheSimulator engine and accounting."""
import pytest
from cache_simulator import CacheSimulator
from policies.lru import LRUPolicy


def test_capacity_one_repeated():
    """Capacity 1 with repeated requests should hit after compulsory miss."""
    sim = CacheSimulator(capacity=1, policy=LRUPolicy())
    res = sim.run([1, 1, 1, 1])
    assert res.total_requests == 4
    assert res.hits == 3
    assert res.misses == 1
    assert res.evictions == 0
    assert res.hit_rate == 0.75
    assert len(sim.cache) == 1


def test_capacity_one_alternating():
    """Capacity 1 with alternating requests causes thrashing: 100% misses."""
    sim = CacheSimulator(capacity=1, policy=LRUPolicy())
    res = sim.run([1, 2, 1, 2, 1, 2])
    assert res.total_requests == 6
    assert res.hits == 0
    assert res.misses == 6
    assert res.evictions == 5
    assert res.hit_rate == 0.0


def test_empty_trace():
    """Empty trace handling produces 0 hits and 0.0 hit rate without errors."""
    sim = CacheSimulator(capacity=5, policy=LRUPolicy())
    res = sim.run([])
    assert res.total_requests == 0
    assert res.hits == 0
    assert res.misses == 0
    assert res.hit_rate == 0.0
    assert res.miss_rate == 0.0


def test_accounting_invariants():
    """Hits + misses must equal total requests, and cache size must not exceed capacity."""
    sim = CacheSimulator(capacity=3, policy=LRUPolicy())
    trace = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
    res = sim.run(trace)

    assert res.hits + res.misses == res.total_requests
    assert res.total_requests == len(trace)
    assert 0.0 <= res.hit_rate <= 1.0
    assert 0.0 <= res.miss_rate <= 1.0
    assert len(sim.cache) <= 3


def test_invalid_capacity():
    """Capacity less than 1 must raise ValueError."""
    with pytest.raises(ValueError):
        CacheSimulator(capacity=0, policy=LRUPolicy())
