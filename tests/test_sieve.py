"""Unit tests for SIEVEPolicy."""
from cache_simulator import CacheSimulator
from policies.sieve import SIEVEPolicy


def test_sieve_basic_operation():
    """Verify SIEVE handles basic hits and misses."""
    policy = SIEVEPolicy(capacity=3)
    sim = CacheSimulator(capacity=3, policy=policy)
    res = sim.run([1, 2, 3, 1, 2])
    assert res.hits == 2
    assert res.misses == 3
    assert len(sim.cache) == 3


def test_sieve_visited_protection():
    """Verify that accessed items (visited=True) survive eviction passes."""
    policy = SIEVEPolicy(capacity=3)
    sim = CacheSimulator(capacity=3, policy=policy)

    # Insert 1, 2, 3 (order in queue: 3 -> 2 -> 1)
    # Access 1 (visited becomes True)
    # Insert 4: hand starts at 1, sees visited=True, sets visited=False, moves to 2 (visited=False)
    # Item 2 must be evicted, not 1!
    res = sim.run([1, 2, 3, 1, 4])
    assert 2 not in sim.cache
    assert 1 in sim.cache
    assert sim.cache == {1, 3, 4}


def test_sieve_respects_capacity():
    """SIEVE cache items must never exceed capacity."""
    policy = SIEVEPolicy(capacity=5)
    sim = CacheSimulator(capacity=5, policy=policy)
    trace = list(range(100))
    sim.run(trace)
    assert len(sim.cache) <= 5
