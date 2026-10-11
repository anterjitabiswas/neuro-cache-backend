"""Unit tests for ARCPolicy."""
from cache_simulator import CacheSimulator
from policies.arc import ARCPolicy


def test_arc_basic_operation():
    """Verify ARC correctly executes basic access sequences."""
    policy = ARCPolicy(capacity=3)
    sim = CacheSimulator(capacity=3, policy=policy)
    # Fill cache: [1, 2, 3]
    res = sim.run([1, 2, 3, 1, 2])
    assert res.hits == 2
    assert res.misses == 3
    assert len(sim.cache) == 3


def test_arc_ghost_b1_adaptation():
    """Verify that a miss hitting B1 increases target parameter p."""
    policy = ARCPolicy(capacity=2)
    sim = CacheSimulator(capacity=2, policy=policy)

    # Access [1, 2] -> in T1.
    # Access 3 -> evicts 1 from T1 into B1.
    sim.run([1, 2, 3])
    assert 1 in policy.b1
    initial_p = policy.p

    # Now request 1 again -> B1 ghost hit!
    sim.step(1)
    # p must have increased
    assert policy.p > initial_p
    assert 1 not in policy.b1
    assert 1 in sim.cache


def test_arc_respects_capacity():
    """ARC cache items must never exceed capacity."""
    policy = ARCPolicy(capacity=4)
    sim = CacheSimulator(capacity=4, policy=policy)
    trace = [1, 2, 3, 4, 5, 2, 6, 1, 7, 8, 3, 4]
    sim.run(trace)
    assert len(sim.cache) <= 4
