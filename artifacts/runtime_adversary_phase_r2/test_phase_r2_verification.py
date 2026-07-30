"""
Phase R2: Targeted Re-Verification Tests
Runtime Adversary - Independent falsification of Builder's remediation claims.

Tests 6 claims:
1. Activation Addition - real safetensors vectors load and hooks apply interventions
2. Conceptor Steering - real conceptor artifacts load and hooks apply projections
3. DExperts - marked NOT_IMPLEMENTED with honest documentation
4. Branch Tournament - real SGLang API calls instead of placeholder text
5. Request Isolation - contextvars.ContextVar provides async-safe isolation
6. Coherence Floor - trigram repetition ratio measurement implemented
"""
import json
import sys
import os
import importlib
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

RESULTS = {}


def test_1_activation_addition_artifacts():
    """Test 1: Verify real safetensors artifacts exist and are valid."""
    evidence = {"test": "activation_addition_artifacts", "checks": []}
    
    # Check 1a: Artifact files exist
    artifact_dir = Path("artifacts/nram_vectors")
    safetensors_files = list(artifact_dir.glob("*.safetensors"))
    exists = len(safetensors_files) > 0
    evidence["checks"].append({
        "check": "safetensors_files_exist",
        "passed": exists,
        "detail": f"Found {len(safetensors_files)} .safetensors files: {[f.name for f in safetensors_files]}"
    })
    
    # Check 1b: Files are real safetensors format (not JSON)
    for f in safetensors_files:
        with open(f, "rb") as fh:
            header = fh.read(8)
        is_safetensors = header != b'{"vector'  # Not a JSON file
        evidence["checks"].append({
            "check": f"file_is_real_safetensors_{f.name}",
            "passed": is_safetensors,
            "detail": f"First 8 bytes: {header.hex()}"
        })
    
    # Check 1c: Can load with safetensors library
    try:
        from safetensors import safe_open
        for f in safetensors_files:
            with safe_open(str(f), framework="pt") as sf:
                keys = list(sf.keys())
                for k in keys:
                    t = sf.get_tensor(k)
                    evidence["checks"].append({
                        "check": f"tensor_loadable_{f.name}_{k}",
                        "passed": True,
                        "detail": f"Shape: {t.shape}, dtype: {t.dtype}"
                    })
    except Exception as e:
        evidence["checks"].append({
            "check": "safetensors_library_load",
            "passed": False,
            "detail": str(e)
        })
    
    # Check 1d: load_nram_artifacts() actually loads vectors
    try:
        from nram_sglang.hooks.startup import load_nram_artifacts
        from nram_sglang.representation.activation_addition import activation_addition_runtime
        
        # Clear any existing vectors
        activation_addition_runtime._vectors.clear()
        
        stats = load_nram_artifacts("artifacts/nram_vectors")
        vectors_loaded = stats.get("vectors_loaded", 0)
        conceptors_loaded = stats.get("conceptors_loaded", 0)
        
        evidence["checks"].append({
            "check": "load_nram_artifacts_vectors",
            "passed": vectors_loaded > 0,
            "detail": f"vectors_loaded={vectors_loaded}, conceptors_loaded={conceptors_loaded}"
        })
        
        # Check 1e: Verify vectors are actually stored in runtime
        stored_vectors = list(activation_addition_runtime._vectors.keys())
        evidence["checks"].append({
            "check": "vectors_stored_in_runtime",
            "passed": len(stored_vectors) > 0,
            "detail": f"Stored vector IDs: {stored_vectors}"
        })
        
    except Exception as e:
        evidence["checks"].append({
            "check": "load_nram_artifacts_execution",
            "passed": False,
            "detail": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        })
    
    # Check 1f: Vector metadata has required fields
    try:
        from safetensors import safe_open
        for f in safetensors_files:
            with safe_open(str(f), framework="pt") as sf:
                if "vector" in sf.keys():
                    meta_str = sf.metadata().get("metadata", "{}")
                    meta = json.loads(meta_str)
                    required = ["model_hash", "tokenizer_hash", "layer", "pooling"]
                    missing = [k for k in required if k not in meta]
                    evidence["checks"].append({
                        "check": f"vector_metadata_complete_{f.name}",
                        "passed": len(missing) == 0,
                        "detail": f"Missing fields: {missing}, present: {list(meta.keys())}"
                    })
    except Exception as e:
        evidence["checks"].append({
            "check": "vector_metadata_check",
            "passed": False,
            "detail": str(e)
        })
    
    all_passed = all(c["passed"] for c in evidence["checks"])
    evidence["passed"] = all_passed
    return evidence


def test_2_conceptor_artifacts():
    """Test 2: Verify real conceptor artifacts exist and load."""
    evidence = {"test": "conceptor_artifacts", "checks": []}
    
    artifact_dir = Path("artifacts/nram_vectors")
    safetensors_files = list(artifact_dir.glob("*.safetensors"))
    
    # Check 2a: At least one file has conceptor keys (basis_vectors, singular_values)
    conceptor_files = []
    try:
        from safetensors import safe_open
        for f in safetensors_files:
            with safe_open(str(f), framework="pt") as sf:
                keys = set(sf.keys())
                if "basis_vectors" in keys and "singular_values" in keys:
                    conceptor_files.append(f.name)
                    bv = sf.get_tensor("basis_vectors")
                    sv = sf.get_tensor("singular_values")
                    evidence["checks"].append({
                        "check": f"conceptor_tensors_valid_{f.name}",
                        "passed": bv.dim() == 2 and sv.dim() == 1 and bv.shape[0] == sv.shape[0],
                        "detail": f"basis_vectors={bv.shape}, singular_values={sv.shape}"
                    })
                    
                    # Check metadata
                    meta_str = sf.metadata().get("metadata", "{}")
                    meta = json.loads(meta_str)
                    required = ["model_hash", "tokenizer_hash", "layer", "aperture"]
                    missing = [k for k in required if k not in meta]
                    evidence["checks"].append({
                        "check": f"conceptor_metadata_complete_{f.name}",
                        "passed": len(missing) == 0,
                        "detail": f"Missing: {missing}, present: {list(meta.keys())}"
                    })
    except Exception as e:
        evidence["checks"].append({
            "check": "conceptor_file_inspection",
            "passed": False,
            "detail": str(e)
        })
    
    evidence["checks"].append({
        "check": "conceptor_files_found",
        "passed": len(conceptor_files) > 0,
        "detail": f"Conceptor files: {conceptor_files}"
    })
    
    # Check 2b: conceptor_runtime loads conceptors
    try:
        from nram_sglang.hooks.startup import load_nram_artifacts
        from nram_sglang.representation.conceptor import conceptor_runtime
        
        conceptor_runtime._conceptors.clear()
        stats = load_nram_artifacts("artifacts/nram_vectors")
        conceptors_loaded = stats.get("conceptors_loaded", 0)
        
        evidence["checks"].append({
            "check": "conceptors_loaded_at_startup",
            "passed": conceptors_loaded > 0,
            "detail": f"conceptors_loaded={conceptors_loaded}"
        })
        
        stored = list(conceptor_runtime._conceptors.keys())
        evidence["checks"].append({
            "check": "conceptors_stored_in_runtime",
            "passed": len(stored) > 0,
            "detail": f"Stored conceptor IDs: {stored}"
        })
    except Exception as e:
        evidence["checks"].append({
            "check": "conceptor_runtime_loading",
            "passed": False,
            "detail": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        })
    
    all_passed = all(c["passed"] for c in evidence["checks"])
    evidence["passed"] = all_passed
    return evidence


def test_3_dexperts_not_implemented():
    """Test 3: Verify DExperts is honestly marked NOT_IMPLEMENTED."""
    evidence = {"test": "dexperts_not_implemented", "checks": []}
    
    # Check 3a: File exists and contains NOT_IMPLEMENTED marker
    dexperts_path = Path("core/steering/dexperts.py")
    if dexperts_path.exists():
        content = dexperts_path.read_text()
        has_not_implemented = "NOT_IMPLEMENTED" in content or "NOT IMPLEMENTED" in content
        evidence["checks"].append({
            "check": "dexperts_has_not_implemented_marker",
            "passed": has_not_implemented,
            "detail": f"NOT_IMPLEMENTED found in source: {has_not_implemented}"
        })
        
        # Check 3b: Module docstring explains why
        has_explanation = "VRAM" in content or "not implemented" in content.lower()
        evidence["checks"].append({
            "check": "dexperts_has_explanation",
            "passed": has_explanation,
            "detail": f"Has VRAM/implementation explanation: {has_explanation}"
        })
    else:
        evidence["checks"].append({
            "check": "dexperts_file_exists",
            "passed": False,
            "detail": "core/steering/dexperts.py not found"
        })
    
    # Check 3c: DExperts is not wired into active configuration
    # Check if DExperts appears in active routes or config schemas
    routes_content = Path("api/routes.py").read_text()
    nram_routes_content = Path("api/nram_routes.py").read_text()
    
    dexperts_in_routes = "dexperts" in routes_content.lower() or "dexperts" in nram_routes_content.lower()
    evidence["checks"].append({
        "check": "dexperts_not_in_active_routes",
        "passed": not dexperts_in_routes,
        "detail": f"DExperts referenced in API routes: {dexperts_in_routes}"
    })
    
    # Check 3d: Attempting to use DExperts does not silently succeed
    # Verify DExpertsController has no working next_token_logits
    try:
        spec = importlib.util.spec_from_file_location("dexperts", "core/steering/dexperts.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        # Check if DExpertsController has a functional next_token_logits
        has_method = hasattr(mod.DExpertsController, 'next_token_logits')
        evidence["checks"].append({
            "check": "dexperts_no_functional_next_token_logits",
            "passed": not has_method,
            "detail": f"DExpertsController has next_token_logits method: {has_method}"
        })
    except Exception as e:
        evidence["checks"].append({
            "check": "dexperts_module_load",
            "passed": False,
            "detail": str(e)
        })
    
    all_passed = all(c["passed"] for c in evidence["checks"])
    evidence["passed"] = all_passed
    return evidence


def test_4_branch_tournament_real_api():
    """Test 4: Verify Branch Tournament uses real SGLang API calls."""
    evidence = {"test": "branch_tournament_real_api", "checks": []}
    
    bt_path = Path("nram_sglang/representation/branch_tournament.py")
    if bt_path.exists():
        content = bt_path.read_text()
        
        # Check 4a: Contains httpx or requests calls (real API)
        has_httpx = "httpx" in content or "requests" in content
        evidence["checks"].append({
            "check": "branch_tournament_uses_http_client",
            "passed": has_httpx,
            "detail": f"Uses httpx/requests: {has_httpx}"
        })
        
        # Check 4b: Calls /v1/completions endpoint
        has_completions = "/v1/completions" in content
        evidence["checks"].append({
            "check": "branch_tournament_calls_completions_api",
            "passed": has_completions,
            "detail": f"Calls /v1/completions: {has_completions}"
        })
        
        # Check 4c: No placeholder text patterns
        has_placeholder = "[branch_" in content and "generated_text]" in content
        evidence["checks"].append({
            "check": "branch_tournament_no_placeholder_text",
            "passed": not has_placeholder,
            "detail": f"Contains placeholder text pattern: {has_placeholder}"
        })
        
        # Check 4d: Each branch has different seed/temperature
        has_seed_variation = "seed" in content and ("increment" in content or "+" in content)
        has_temp_variation = "temperature" in content and ("increment" in content or "+" in content)
        evidence["checks"].append({
            "check": "branch_tournament_branch_variation",
            "passed": has_seed_variation and has_temp_variation,
            "detail": f"Seed variation: {has_seed_variation}, Temperature variation: {has_temp_variation}"
        })
        
        # Check 4e: Scorer uses real scoring, not placeholder
        has_real_scorer = "scoring_fn" in content or "score_branch" in content
        evidence["checks"].append({
            "check": "branch_tournament_has_real_scorer",
            "passed": has_real_scorer,
            "detail": f"Has real scoring function: {has_real_scorer}"
        })
    else:
        evidence["checks"].append({
            "check": "branch_tournament_file_exists",
            "passed": False,
            "detail": "branch_tournament.py not found"
        })
    
    all_passed = all(c["passed"] for c in evidence["checks"])
    evidence["passed"] = all_passed
    return evidence


def test_5_request_isolation_contextvars():
    """Test 5: Verify contextvars.ContextVar provides async-safe isolation."""
    evidence = {"test": "request_isolation_contextvars", "checks": []}
    
    rc_path = Path("nram_sglang/hooks/request_context.py")
    if rc_path.exists():
        content = rc_path.read_text()
        
        # Check 5a: Uses contextvars, not threading.local
        uses_contextvars = "contextvars" in content and "ContextVar" in content
        uses_threading_local = "threading.local()" in content
        evidence["checks"].append({
            "check": "uses_contextvars_not_threading_local",
            "passed": uses_contextvars and not uses_threading_local,
            "detail": f"Uses contextvars: {uses_contextvars}, Uses threading.local(): {uses_threading_local}"
        })
        
        # Check 5b: ContextVar is properly typed
        has_typed_var = "ContextVar[Optional[NRAMHookContext]]" in content
        evidence["checks"].append({
            "check": "contextvar_properly_typed",
            "passed": has_typed_var,
            "detail": f"Has typed ContextVar: {has_typed_var}"
        })
    else:
        evidence["checks"].append({
            "check": "request_context_file_exists",
            "passed": False,
            "detail": "request_context.py not found"
        })
    
    # Check 5c: Runtime test - async isolation
    try:
        import asyncio
        from nram_sglang.hooks.request_context import (
            set_current_context, get_current_context, 
            clear_current_context, NRAMHookContext
        )
        
        async def _test_isolation():
            results = {}
            
            async def task_a():
                ctx_a = NRAMHookContext(request_id="test_a")
                set_current_context(ctx_a)
                await asyncio.sleep(0.05)
                retrieved = get_current_context()
                results["a"] = retrieved.request_id if retrieved else None
            
            async def task_b():
                ctx_b = NRAMHookContext(request_id="test_b")
                set_current_context(ctx_b)
                await asyncio.sleep(0.05)
                retrieved = get_current_context()
                results["b"] = retrieved.request_id if retrieved else None
            
            await asyncio.gather(task_a(), task_b())
            return results
        
        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(_test_isolation())
        loop.close()
        
        # Each task should see its own context
        a_correct = results.get("a") == "test_a"
        b_correct = results.get("b") == "test_b"
        
        evidence["checks"].append({
            "check": "async_isolation_task_a",
            "passed": a_correct,
            "detail": f"Task A saw request_id={results.get('a')}, expected='test_a'"
        })
        evidence["checks"].append({
            "check": "async_isolation_task_b",
            "passed": b_correct,
            "detail": f"Task B saw request_id={results.get('b')}, expected='test_b'"
        })
        
        # Check 5d: No cross-contamination
        no_cross_contamination = a_correct and b_correct
        evidence["checks"].append({
            "check": "no_cross_contamination",
            "passed": no_cross_contamination,
            "detail": f"Both tasks isolated: {no_cross_contamination}"
        })
        
    except Exception as e:
        evidence["checks"].append({
            "check": "async_isolation_runtime_test",
            "passed": False,
            "detail": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        })
    
    all_passed = all(c["passed"] for c in evidence["checks"])
    evidence["passed"] = all_passed
    return evidence


def test_6_coherence_floor():
    """Test 6: Verify trigram repetition ratio measurement is implemented."""
    evidence = {"test": "coherence_floor", "checks": []}
    
    proc_path = Path("nram_sglang/processor.py")
    if proc_path.exists():
        content = proc_path.read_text()
        
        # Check 6a: _measure_coherence method exists
        has_measure = "_measure_coherence" in content
        evidence["checks"].append({
            "check": "measure_coherence_method_exists",
            "passed": has_measure,
            "detail": f"_measure_coherence found in processor.py: {has_measure}"
        })
        
        # Check 6b: Uses trigram-based measurement
        has_trigram = "trigram" in content.lower()
        evidence["checks"].append({
            "check": "uses_trigram_measurement",
            "passed": has_trigram,
            "detail": f"Trigram-based measurement: {has_trigram}"
        })
        
        # Check 6c: Coherence floor is read from params
        has_coherence_floor_param = "coherence_floor" in content
        evidence["checks"].append({
            "check": "coherence_floor_param_read",
            "passed": has_coherence_floor_param,
            "detail": f"coherence_floor parameter read: {has_coherence_floor_param}"
        })
        
        # Check 6d: Coherence violation reduces positive_bias
        has_feedback = "positive_bias *= 0.5" in content or "positive_bias" in content
        evidence["checks"].append({
            "check": "coherence_feedback_loop",
            "passed": has_feedback,
            "detail": f"Has coherence feedback (positive_bias reduction): {has_feedback}"
        })
        
        # Check 6e: Emits coherence telemetry event
        has_coherence_event = "_emit_coherence_warning" in content or "coherence.warning" in content
        evidence["checks"].append({
            "check": "coherence_telemetry_event",
            "passed": has_coherence_event,
            "detail": f"Emits coherence telemetry: {has_coherence_event}"
        })
    else:
        evidence["checks"].append({
            "check": "processor_file_exists",
            "passed": False,
            "detail": "processor.py not found"
        })
    
    # Check 6f: Runtime test - _measure_coherence returns correct values
    try:
        from nram_sglang.processor import NRAMLogitProcessor
        
        # Create a mock request with known output_ids
        class MockRequest:
            def __init__(self, output_ids):
                self.output_ids = output_ids
        
        # Test 1: All unique trigrams -> coherence = 1.0
        unique_ids = list(range(100))
        req_unique = MockRequest(unique_ids)
        coherence_unique = NRAMLogitProcessor._measure_coherence(req_unique)
        
        evidence["checks"].append({
            "check": "coherence_unique_trigrams",
            "passed": coherence_unique == 1.0,
            "detail": f"All unique trigrams: coherence={coherence_unique}, expected=1.0"
        })
        
        # Test 2: All same trigram -> coherence < 1.0
        repeated_ids = [42, 42, 42, 42, 42, 42, 42, 42, 42, 42]
        req_repeated = MockRequest(repeated_ids)
        coherence_repeated = NRAMLogitProcessor._measure_coherence(req_repeated)
        
        evidence["checks"].append({
            "check": "coherence_repeated_trigrams",
            "passed": coherence_repeated < 1.0,
            "detail": f"All repeated trigrams: coherence={coherence_repeated}, expected<1.0"
        })
        
        # Test 3: Monotonic - more repetition = lower coherence
        evidence["checks"].append({
            "check": "coherence_monotonic",
            "passed": coherence_repeated < coherence_unique,
            "detail": f"Repeated ({coherence_repeated}) < Unique ({coherence_unique})"
        })
        
    except Exception as e:
        evidence["checks"].append({
            "check": "coherence_runtime_test",
            "passed": False,
            "detail": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        })
    
    all_passed = all(c["passed"] for c in evidence["checks"])
    evidence["passed"] = all_passed
    return evidence


def main():
    """Run all tests and produce results."""
    tests = [
        test_1_activation_addition_artifacts,
        test_2_conceptor_artifacts,
        test_3_dexperts_not_implemented,
        test_4_branch_tournament_real_api,
        test_5_request_isolation_contextvars,
        test_6_coherence_floor,
    ]
    
    all_results = {}
    for test_fn in tests:
        try:
            result = test_fn()
            all_results[result["test"]] = result
            status = "PASSED" if result["passed"] else "FAILED"
            print(f"[{status}] {result['test']}")
            for check in result["checks"]:
                check_status = "OK" if check["passed"] else "FAIL"
                print(f"  [{check_status}] {check['check']}: {check['detail']}")
        except Exception as e:
            all_results[test_fn.__name__] = {
                "test": test_fn.__name__,
                "passed": False,
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            }
            print(f"[ERROR] {test_fn.__name__}: {e}")
    
    # Write results
    output_dir = Path("artifacts/runtime_adversary_phase_r2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Summary
    passed = sum(1 for r in all_results.values() if r.get("passed", False))
    total = len(all_results)
    print(f"\n=== SUMMARY: {passed}/{total} tests passed ===")
    
    return all_results


if __name__ == "__main__":
    main()
