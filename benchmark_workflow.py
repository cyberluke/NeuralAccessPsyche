"""Benchmark script for workflow execution performance.

Measures:
- Workflow creation time
- Event logging throughput
- Artifact storage performance
- Workflow retrieval and listing performance
- SQLite query performance under load
"""
import time
import statistics
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.agentic.workflow_store import WorkflowStore
from core.agentic.contracts import (
    InnovationWorkflowState,
    WorkflowStatus,
    WorkflowEvent,
)
from datetime import datetime, timezone


def _make_state(workflow_id, phase="repository_ingestion", status=WorkflowStatus.RUNNING):
    """Helper to create a valid InnovationWorkflowState."""
    return InnovationWorkflowState(
        workflow_id=workflow_id,
        repository_path="/test/repo",
        repository_revision="abc123",
        user_goal="Benchmark goal",
        status=status,
        phase=phase,
    )


def benchmark_workflow_creation(store, num_workflows=100):
    """Benchmark workflow creation performance."""
    times = []
    
    for i in range(num_workflows):
        state = _make_state(f"wf-bench-{i}")
        
        start = time.perf_counter()
        store.create_workflow(state)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "total_ms": sum(times),
        "throughput": num_workflows / (sum(times) / 1000),
    }


def benchmark_event_logging(store, workflow_id, num_events=100):
    """Benchmark event logging performance."""
    times = []
    
    for seq in range(1, num_events + 1):
        event = WorkflowEvent(
            workflow_id=workflow_id,
            sequence=seq,
            event_type="benchmark_event",
            phase="benchmark_phase",
            summary=f"Benchmark event {seq}",
            timestamp=datetime.now(timezone.utc),
        )
        
        start = time.perf_counter()
        store.add_event(event)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "total_ms": sum(times),
        "throughput": num_events / (sum(times) / 1000),
    }


def benchmark_artifact_storage(store, workflow_id, num_artifacts=50):
    """Benchmark artifact storage performance."""
    times = []
    
    for i in range(num_artifacts):
        if i % 2 == 0:
            artifact_data = {
                "direction": f"Direction {i}",
                "rationale": f"Rationale {i}",
                "evidence_ids": [f"ev-{j}" for j in range(5)],
            }
            artifact_type = "selected_direction"
        else:
            artifact_data = {
                "phases": [{"phase": f"Phase {j}", "tasks": [f"Task {k}" for k in range(3)]} for j in range(5)],
                "evidence_ids": [f"ev-{j}" for j in range(5)],
            }
            artifact_type = "roadmap"
        
        start = time.perf_counter()
        store.store_artifact(workflow_id, f"art-{i}", artifact_type, artifact_data)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "total_ms": sum(times),
        "throughput": num_artifacts / (sum(times) / 1000),
    }


def benchmark_workflow_retrieval(store, num_workflows=100):
    """Benchmark workflow retrieval performance."""
    times = []
    
    for i in range(num_workflows):
        start = time.perf_counter()
        workflow = store.get_workflow(f"wf-bench-{i}")
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "total_ms": sum(times),
    }


def benchmark_workflow_listing(store, num_iterations=10):
    """Benchmark workflow listing performance."""
    times = []
    
    for _ in range(num_iterations):
        start = time.perf_counter()
        workflows = store.list_workflows()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "total_workflows": len(workflows),
    }


def benchmark_event_retrieval(store, workflow_id, num_iterations=10):
    """Benchmark event retrieval performance."""
    times = []
    
    for _ in range(num_iterations):
        start = time.perf_counter()
        events = store.get_events(workflow_id)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "total_events": len(events),
    }


def benchmark_artifact_retrieval(store, workflow_id, num_iterations=10):
    """Benchmark artifact retrieval performance."""
    times = []
    
    for _ in range(num_iterations):
        start = time.perf_counter()
        artifacts = store.get_artifacts_by_type(workflow_id, "selected_direction")
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "total_artifacts": len(artifacts),
    }


def main():
    """Run all workflow benchmarks."""
    print("=" * 70)
    print("Workflow Store Performance Benchmark")
    print("=" * 70)
    print()
    
    # Create temporary database
    temp_dir = tempfile.mkdtemp()
    db_path = str(Path(temp_dir) / "benchmark_workflows.db")
    store = WorkflowStore(db_path=db_path)
    
    num_workflows = 100
    num_events = 100
    num_artifacts = 50
    
    print(f"Configuration:")
    print(f"  Workflows: {num_workflows}")
    print(f"  Events per workflow: {num_events}")
    print(f"  Artifacts per workflow: {num_artifacts}")
    print()
    
    # Workflow creation
    print("1. Workflow Creation")
    print("-" * 70)
    creation_stats = benchmark_workflow_creation(store, num_workflows)
    print(f"  Mean:        {creation_stats['mean_ms']:.3f} ms")
    print(f"  Median:      {creation_stats['median_ms']:.3f} ms")
    print(f"  Std:         {creation_stats['std_ms']:.3f} ms")
    print(f"  Total:       {creation_stats['total_ms']:.1f} ms")
    print(f"  Throughput:  {creation_stats['throughput']:.1f} workflows/s")
    print()
    
    # Event logging
    print("2. Event Logging")
    print("-" * 70)
    event_stats = benchmark_event_logging(store, "wf-bench-0", num_events)
    print(f"  Mean:        {event_stats['mean_ms']:.3f} ms")
    print(f"  Median:      {event_stats['median_ms']:.3f} ms")
    print(f"  Std:         {event_stats['std_ms']:.3f} ms")
    print(f"  Total:       {event_stats['total_ms']:.1f} ms")
    print(f"  Throughput:  {event_stats['throughput']:.1f} events/s")
    print()
    
    # Artifact storage
    print("3. Artifact Storage")
    print("-" * 70)
    artifact_stats = benchmark_artifact_storage(store, "wf-bench-0", num_artifacts)
    print(f"  Mean:        {artifact_stats['mean_ms']:.3f} ms")
    print(f"  Median:      {artifact_stats['median_ms']:.3f} ms")
    print(f"  Std:         {artifact_stats['std_ms']:.3f} ms")
    print(f"  Total:       {artifact_stats['total_ms']:.1f} ms")
    print(f"  Throughput:  {artifact_stats['throughput']:.1f} artifacts/s")
    print()
    
    # Workflow retrieval
    print("4. Workflow Retrieval")
    print("-" * 70)
    retrieval_stats = benchmark_workflow_retrieval(store, num_workflows)
    print(f"  Mean:        {retrieval_stats['mean_ms']:.3f} ms")
    print(f"  Median:      {retrieval_stats['median_ms']:.3f} ms")
    print(f"  Std:         {retrieval_stats['std_ms']:.3f} ms")
    print(f"  Total:       {retrieval_stats['total_ms']:.1f} ms")
    print()
    
    # Workflow listing
    print("5. Workflow Listing")
    print("-" * 70)
    listing_stats = benchmark_workflow_listing(store, 10)
    print(f"  Mean:        {listing_stats['mean_ms']:.3f} ms")
    print(f"  Median:      {listing_stats['median_ms']:.3f} ms")
    print(f"  Std:         {listing_stats['std_ms']:.3f} ms")
    print(f"  Workflows:   {listing_stats['total_workflows']}")
    print()
    
    # Event retrieval
    print("6. Event Retrieval")
    print("-" * 70)
    event_retrieval_stats = benchmark_event_retrieval(store, "wf-bench-0", 10)
    print(f"  Mean:        {event_retrieval_stats['mean_ms']:.3f} ms")
    print(f"  Median:      {event_retrieval_stats['median_ms']:.3f} ms")
    print(f"  Std:         {event_retrieval_stats['std_ms']:.3f} ms")
    print(f"  Events:      {event_retrieval_stats['total_events']}")
    print()
    
    # Artifact retrieval
    print("7. Artifact Retrieval")
    print("-" * 70)
    artifact_retrieval_stats = benchmark_artifact_retrieval(store, "wf-bench-0", 10)
    print(f"  Mean:        {artifact_retrieval_stats['mean_ms']:.3f} ms")
    print(f"  Median:      {artifact_retrieval_stats['median_ms']:.3f} ms")
    print(f"  Std:         {artifact_retrieval_stats['std_ms']:.3f} ms")
    print(f"  Artifacts:   {artifact_retrieval_stats['total_artifacts']}")
    print()
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
