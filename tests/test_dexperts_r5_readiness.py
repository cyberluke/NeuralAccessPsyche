from scripts.dexperts_r5_batch_parity import compare
from scripts.dexperts_r5_batch_runner import length_buckets


def _rows():
    return [{
        "sequence_id": f"s{i:02d}", "prompt": f"prompt {i}",
        "source_tokens": (i % 4) + 1,
        "target_token_delta_logp": 0.1, "lr": 0.2,
        "kl": 0.01, "js": 0.02, "tv": 0.03,
        "control_verdict": "pass",
    } for i in range(16)]


def test_length_buckets_respect_size_and_token_budget():
    batches = list(length_buckets(_rows(), bucket_size=8, token_budget=20))
    assert all(len(batch) <= 8 for batch in batches)
    assert all(sum(row["source_tokens"] for row in batch) <= 20 for batch in batches)
    assert sorted(row["sequence_id"] for batch in batches for row in batch) == [f"s{i:02d}" for i in range(16)]


def test_batch_parity_passes_within_tolerance():
    result = compare(_rows(), _rows(), tolerance=1e-3)
    assert result["passed"]
    assert result["ordering_identical"]
    assert result["controls_identical"]
    assert not result["cross_request_contamination"]


def test_batch_parity_rejects_metric_drift():
    batch8 = _rows()
    batch8[7]["kl"] = 1.0
    result = compare(_rows(), batch8, tolerance=1e-3)
    assert not result["passed"]
    assert result["max_abs_difference"]["kl"] > 1e-3
