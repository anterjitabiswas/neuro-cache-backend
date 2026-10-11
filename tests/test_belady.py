"""Unit tests for Belady's Optimal offline oracle."""
from cache_simulator import CacheSimulator
from policies.belady import BeladyPolicy
from policies.lru import LRUPolicy
from policies.lfu import LFUPolicy


def test_belady_evicts_furthest_future_use():
    """Verify that Belady chooses the item accessed furthest in future."""
    # Trace: 1, 2, 3, 4, 1, 2
    # At step 3 (when 4 arrives), cache has {1, 2, 3}.
    # Next uses: 1 is at step 4, 2 is at step 5, 3 is never used again!
    # Belady must evict 3!
    trace = [1, 2, 3, 4, 1, 2]
    policy = BeladyPolicy(trace=trace)
    sim = CacheSimulator(capacity=3, policy=policy)
    sim.run(trace)
    # Final cache should contain 1, 2, 4 (3 was evicted)
    assert 3 not in sim.cache
    assert sim.cache == {1, 2, 4}


def test_belady_optimality_upper_bound():
    """Belady's hit rate must be greater than or equal to both LRU and LFU."""
    trace = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 4, 5]
    capacity = 3

    belady_res = CacheSimulator(capacity=capacity, policy=BeladyPolicy(trace=trace)).run(trace)
    lru_res = CacheSimulator(capacity=capacity, policy=LRUPolicy()).run(trace)
    lfu_res = CacheSimulator(capacity=capacity, policy=LFUPolicy()).run(trace)

    assert belady_res.hit_rate >= lru_res.hit_rate
    assert belady_res.hit_rate >= lfu_res.hit_rate
