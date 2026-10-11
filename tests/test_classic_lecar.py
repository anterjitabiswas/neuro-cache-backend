"""Unit tests for ClassicLeCaRPolicy."""
from cache_simulator import CacheSimulator
from policies.classic_lecar import ClassicLeCaRPolicy


def test_classic_lecar_operation():
    """Verify classic LeCaR runs and maintains capacity."""
    policy = ClassicLeCaRPolicy(capacity=4)
    sim = CacheSimulator(capacity=4, policy=policy)
    res = sim.run([1, 2, 3, 4, 1, 5, 2, 6])
    assert res.hits >= 1
    assert len(sim.cache) <= 4
