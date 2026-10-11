"""Unit tests for LRUPolicy."""
from cache_simulator import CacheSimulator
from policies.lru import LRUPolicy


def test_lru_basic_eviction():
    """Verify that LRU evicts the least recently accessed item."""
    policy = LRUPolicy()
    sim = CacheSimulator(capacity=3, policy=policy)
    # [1, 2, 3] fills cache
    # 4 evicts 1 (least recent) -> cache is {2, 3, 4}
    res = sim.run([1, 2, 3, 4])
    assert sim.cache == {2, 3, 4}
    assert res.hits == 0
    assert res.misses == 4
    assert res.evictions == 1


def test_lru_recency_update_on_hit():
    """Verify that hits move an item to MRU, protecting it from eviction."""
    policy = LRUPolicy()
    sim = CacheSimulator(capacity=3, policy=policy)
    # 1, 2, 3 inserted
    # 1 is accessed again (becomes MRU) -> recency order: 2 (LRU), 3, 1 (MRU)
    # 4 is inserted -> 2 must be evicted, not 1!
    res = sim.run([1, 2, 3, 1, 4])
    assert 2 not in sim.cache
    assert 1 in sim.cache
    assert sim.cache == {1, 3, 4}
    assert res.hits == 1
    assert res.evictions == 1
