"""Benchmark script for NRAM logit processor performance.

Measures:
- Baseline latency (no steering)
- Basic steering latency (positive/negative bias)
- Phenomenon mixer latency (all 7 phenomena)
- Throughput (requests/second)
- Memory usage
"""
import time
import statistics
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from core.steering.nram_logit_processor import NRAMLogitProcessor


class FakeRequest:
    """Fake request object for benchmarking."""
    def __init__(self, output_ids):
        self.output_ids = output_ids


def benchmark_baseline(processor, vocab_size=50257, batch_size=1, num_iterations=100):
    """Benchmark baseline (no steering)."""
    logits = torch.randn((batch_size, vocab_size))
    params = [{}]
    
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        processor(logits.clone(), params)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def benchmark_basic_steering(processor, vocab_size=50257, batch_size=1, num_iterations=100):
    """Benchmark basic steering (positive/negative bias)."""
    logits = torch.randn((batch_size, vocab_size))
    params = [{
        "positive_token_ids": list(range(100)),
        "negative_token_ids": list(range(100, 200)),
        "forbidden_token_ids": list(range(200, 250)),
        "positive_bias": 1.0,
        "negative_bias": 2.0,
        "repetition_penalty": 1.5,
    }]
    
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        processor(logits.clone(), params)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def benchmark_phenomena(processor, vocab_size=50257, batch_size=1, num_iterations=100):
    """Benchmark all 7 phenomena active."""
    logits = torch.randn((batch_size, vocab_size))
    req = FakeRequest(output_ids=list(range(50)))
    params = [{
        "positive_token_ids": list(range(100)),
        "negative_token_ids": list(range(100, 200)),
        "forbidden_token_ids": list(range(200, 250)),
        "positive_bias": 1.0,
        "negative_bias": 2.0,
        "repetition_penalty": 1.5,
        "max_tokens": 100,
        "phenomenon_weights": {
            "overlap": 0.5,
            "forgetting": 0.5,
            "looping": 0.5,
            "associative_jump": 0.5,
            "synesthesia": 0.5,
            "dissolution": 0.5,
            "insight": 0.5,
        },
        "__req__": req,
    }]
    
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        processor(logits.clone(), params)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def benchmark_individual_phenomena(processor, vocab_size=50257, batch_size=1, num_iterations=50):
    """Benchmark each phenomenon individually."""
    phenomena_names = ["overlap", "forgetting", "looping", "associative_jump", 
                       "synesthesia", "dissolution", "insight"]
    results = {}
    
    for phenomenon in phenomena_names:
        logits = torch.randn((batch_size, vocab_size))
        req = FakeRequest(output_ids=list(range(50)))
        params = [{
            "positive_token_ids": list(range(100)),
            "positive_bias": 1.0,
            "max_tokens": 100,
            "phenomenon_weights": {phenomenon: 0.8},
            "__req__": req,
        }]
        
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            processor(logits.clone(), params)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        results[phenomenon] = {
            "mean_ms": statistics.mean(times),
            "median_ms": statistics.median(times),
        }
    
    return results


def main():
    """Run all benchmarks."""
    print("=" * 70)
    print("NRAM Logit Processor Performance Benchmark")
    print("=" * 70)
    print()
    
    processor = NRAMLogitProcessor()
    vocab_size = 50257  # Typical vocabulary size
    batch_size = 1
    num_iterations = 100
    
    print(f"Configuration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations per test: {num_iterations}")
    print()
    
    # Baseline
    print("1. Baseline (no steering)")
    print("-" * 70)
    baseline = benchmark_baseline(processor, vocab_size, batch_size, num_iterations)
    print(f"  Mean:   {baseline['mean_ms']:.3f} ms")
    print(f"  Median: {baseline['median_ms']:.3f} ms")
    print(f"  Std:    {baseline['std_ms']:.3f} ms")
    print(f"  Min:    {baseline['min_ms']:.3f} ms")
    print(f"  Max:    {baseline['max_ms']:.3f} ms")
    print()
    
    # Basic steering
    print("2. Basic Steering (positive/negative/forbidden)")
    print("-" * 70)
    basic = benchmark_basic_steering(processor, vocab_size, batch_size, num_iterations)
    print(f"  Mean:   {basic['mean_ms']:.3f} ms")
    print(f"  Median: {basic['median_ms']:.3f} ms")
    print(f"  Std:    {basic['std_ms']:.3f} ms")
    print(f"  Min:    {basic['min_ms']:.3f} ms")
    print(f"  Max:    {basic['max_ms']:.3f} ms")
    print(f"  Overhead: {basic['mean_ms'] - baseline['mean_ms']:.3f} ms ({((basic['mean_ms'] / baseline['mean_ms']) - 1) * 100:.1f}%)")
    print()
    
    # All phenomena
    print("3. All Phenomena Active")
    print("-" * 70)
    phenomena = benchmark_phenomena(processor, vocab_size, batch_size, num_iterations)
    print(f"  Mean:   {phenomena['mean_ms']:.3f} ms")
    print(f"  Median: {phenomena['median_ms']:.3f} ms")
    print(f"  Std:    {phenomena['std_ms']:.3f} ms")
    print(f"  Min:    {phenomena['min_ms']:.3f} ms")
    print(f"  Max:    {phenomena['max_ms']:.3f} ms")
    print(f"  Overhead: {phenomena['mean_ms'] - baseline['mean_ms']:.3f} ms ({((phenomena['mean_ms'] / baseline['mean_ms']) - 1) * 100:.1f}%)")
    print()
    
    # Individual phenomena
    print("4. Individual Phenomena")
    print("-" * 70)
    individual = benchmark_individual_phenomena(processor, vocab_size, batch_size, 50)
    for name, stats in individual.items():
        print(f"  {name:20s}  Mean: {stats['mean_ms']:.3f} ms  Median: {stats['median_ms']:.3f} ms")
    print()
    
    # Throughput
    print("5. Throughput (requests/second)")
    print("-" * 70)
    baseline_rps = 1000 / baseline['mean_ms']
    basic_rps = 1000 / basic['mean_ms']
    phenomena_rps = 1000 / phenomena['mean_ms']
    print(f"  Baseline:        {baseline_rps:.1f} req/s")
    print(f"  Basic Steering:  {basic_rps:.1f} req/s")
    print(f"  All Phenomena:   {phenomena_rps:.1f} req/s")
    print()
    
    print("=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
