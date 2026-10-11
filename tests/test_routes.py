"""Integration tests for SMARTCACHE AI FastAPI routes."""
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["Project"] == "Smartcache-ai"
    assert data["status"] == "Running"


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "Ok"}


def test_get_policies():
    response = client.get("/policies")
    assert response.status_code == 200
    data = response.json()
    assert "policies" in data
    assert "LRU" in data["policies"]
    assert "LFU" in data["policies"]
    assert "Belady's Algorithm" in data["policies"]
    assert "Smartcache Policy" in data["policies"]


def test_get_workloads():
    response = client.get("/workloads")
    assert response.status_code == 200
    data = response.json()
    assert "workloads" in data
    assert "Looping hot set" in data["workloads"]
    assert "Sequential scan" in data["workloads"]
    assert "Zipf-like skew" in data["workloads"]
    assert "Phase shift" in data["workloads"]
    assert "Mixed workload" in data["workloads"]


def test_simulation_execution():
    payload = {
        "workload_type": "Looping hot set",
        "cache_capacity": 10,
        "trace_length": 200,
        "policy": "Smartcache Policy",
        "random_seed": 42,
    }
    response = client.post("/simulations", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Simulation completed successfully"
    assert data["configuration"]["cache_capacity"] == 10
    assert "simulation_result" in data
    sim = data["simulation_result"]
    assert sim["total_requests"] == 200
    assert sim["hits"] + sim["misses"] == 200
    assert sim["cache_hits"] == sim["hits"]
    assert sim["cache_misses"] == sim["misses"]
    assert 0.0 <= sim["hit_rate"] <= 1.0
    assert 0.0 <= sim["miss_rate"] <= 1.0
    assert "wall_clock_time_sec" in sim
    assert "compulsory_misses" in sim


def test_simulation_with_events():
    payload = {
        "workload_type": "Sequential scan",
        "cache_capacity": 5,
        "trace_length": 50,
        "policy": "LRU",
        "random_seed": 42,
        "include_events": True,
    }
    response = client.post("/simulations", json=payload)
    assert response.status_code == 200
    data = response.json()
    sim = data["simulation_result"]
    assert "events" in sim
    assert len(sim["events"]) == 50
    assert "rolling_hit_rate" in sim
    assert len(sim["rolling_hit_rate"]) == 50


def test_simulation_invalid_capacity():
    payload = {
        "workload_type": "Looping hot set",
        "cache_capacity": 0,
        "trace_length": 100,
        "policy": "LRU",
    }
    response = client.post("/simulations", json=payload)
    assert response.status_code == 400
    assert "positive integers" in response.json()["detail"]


def test_simulation_invalid_trace_length():
    payload = {
        "workload_type": "Looping hot set",
        "cache_capacity": 10,
        "trace_length": -10,
        "policy": "LRU",
    }
    response = client.post("/simulations", json=payload)
    assert response.status_code == 400


def test_simulation_invalid_policy():
    payload = {
        "workload_type": "Looping hot set",
        "cache_capacity": 10,
        "trace_length": 100,
        "policy": "InvalidPolicyXYZ",
    }
    response = client.post("/simulations", json=payload)
    assert response.status_code == 400
    assert "Unsupported policy" in response.json()["detail"]


def test_simulation_invalid_workload():
    payload = {
        "workload_type": "InvalidWorkloadXYZ",
        "cache_capacity": 10,
        "trace_length": 100,
        "policy": "LRU",
    }
    response = client.post("/simulations", json=payload)
    assert response.status_code == 400
    assert "Unsupported workload_type" in response.json()["detail"]


def test_compare_simulations_success():
    payload = {
        "workload_type": "Phase shift",
        "cache_capacity": 10,
        "trace_length": 300,
        "policy": ["LRU", "LFU", "Belady's Algorithm", "Smartcache Policy"],
        "random_seed": 42,
    }
    response = client.post("/simulations/compare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Comparison successful"
    results = data["comparison_results"]
    for pol in ["LRU", "LFU", "Belady's Algorithm", "Smartcache Policy"]:
        assert pol in results
        assert results[pol]["total_requests"] == 300
        assert 0.0 <= results[pol]["hit_rate"] <= 1.0


def test_compare_simulations_empty_policy_list():
    payload = {
        "workload_type": "Looping hot set",
        "cache_capacity": 10,
        "trace_length": 100,
        "policy": [],
    }
    response = client.post("/simulations/compare", json=payload)
    assert response.status_code == 400
