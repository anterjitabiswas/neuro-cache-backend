"""Unit tests for workload generators and file loader."""
from workloads.generators import (
    generate_looping_trace,
    generate_sequential_scan_trace,
    generate_zipf_trace,
    generate_phase_shift_trace,
    generate_mixed_trace,
)
from workloads.loaders import load_trace_from_file


def test_generators_deterministic_seed():
    """Verify that all generators produce identical output given the same seed."""
    t1 = generate_looping_trace(seed=42)
    t2 = generate_looping_trace(seed=42)
    assert t1 == t2

    z1 = generate_zipf_trace(seed=42)
    z2 = generate_zipf_trace(seed=42)
    assert z1 == z2

    p1 = generate_phase_shift_trace(seed=42)
    p2 = generate_phase_shift_trace(seed=42)
    assert p1 == p2

    m1 = generate_mixed_trace(seed=42)
    m2 = generate_mixed_trace(seed=42)
    assert m1 == m2


def test_sequential_scan():
    """Verify sequential scan output format."""
    s = generate_sequential_scan_trace(scan_length=50)
    assert len(s) == 50
    assert s == list(range(50))


def test_file_loader(tmp_path):
    """Verify file loader handles line, comma, and space delimiters."""
    file = tmp_path / "test_trace.txt"
    file.write_text("1, 2, 3\n4 5 6\n7\n8,9", encoding="utf-8")

    loaded = load_trace_from_file(file)
    assert loaded == [1, 2, 3, 4, 5, 6, 7, 8, 9]
