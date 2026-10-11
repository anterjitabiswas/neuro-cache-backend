"""Unit tests for LFUPolicy with deterministic LRU tie-breaking."""
from cache_simulator import CacheSimulator
from policies.lfu import LFUPolicy


def test_lfu_evicts_lowest_frequency():
    """Verify that item with lowest access frequency is evicted."""
    policy = LFUPolicy()
    sim = CacheSimulator(capacity=3, policy=policy)
    # 1 accessed twice (f=2), 2 accessed twice (f=2), 3 accessed once (f=1)
    # 4 inserted -> 3 must be evicted!
    res = sim.run([1, 1, 2, 2, 3, 4])
    assert 3 not in sim.cache
    assert sim.cache == {1, 2, 4}
    assert res.hits == 2
    assert res.evictions == 1


def test_lfu_tie_breaker_lru():
    """Verify that when frequencies are tied, the least recently used is evicted."""
    policy = LFUPolicy()
    sim = CacheSimulator(capacity=3, policy=policy)
    # 1, 2, 3 all have freq=1, access order: 1, 2, 3
    # 4 inserted -> 1 must be evicted
    sim.run([1, 2, 3, 4])
    assert 1 not in sim.cache
    assert sim.cache == {2, 3, 4}

    # Now access 2 -> freq(2)=2. freq(3)=1, freq(4)=1
    # 5 inserted -> 3 has freq 1 and was accessed before 4, so 3 must be evicted!
    sim.run([1, 2, 3, 4, 2, 5])
    assert 3 not in sim.cache
    assert sim.cache == {2, 4, 5}
