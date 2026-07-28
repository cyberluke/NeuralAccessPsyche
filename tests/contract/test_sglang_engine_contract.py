"""Phase 10-12: SGLang engine contract tests (no GPU required).

Validates the security boundary, alias routing, allowlist filtering,
and NRAM processor injection at the payload level.
"""
import pytest

from core.contracts.openai import ChatCompletionRequest
from core.engines.sglang_engine import (
    SGLangEngine,
    SGLangEngineError,
    MODEL_ALIAS_MAP,
    NRAM_ENABLED_ALIASES,
    _FORBIDDEN_CLIENT_FIELDS,
)


class FakeTokenizer:
    vocab_size = 100
    _enc = {" human": [11], " synergy": [31]}
    _dec = {11: " human", 31: " synergy"}

    def encode(self, text, add_special_tokens=False):
        return list(self._enc.get(text, []))

    def decode(self, ids):
        return "".join(self._dec.get(i, "?") for i in ids)

    def __len__(self):
        return self.vocab_size


@pytest.fixture
def engine():
    """Engine constructed with a fake tokenizer; no network calls made here."""
    return SGLangEngine(
        base_url="http://sglang:30000/v1",
        model="nram-qwen3-14b-awq",
        tokenizer=FakeTokenizer(),
    )


def make_request(model, nram=None, **kw):
    return ChatCompletionRequest(
        model=model,
        messages=[{"role": "user", "content": "Hello"}],
        nram=nram,
        **kw,
    )


class TestAliasRouting:
    def test_both_aliases_map_to_same_upstream(self):
        assert MODEL_ALIAS_MAP["nram-qwen3-14b-awq"] == "nram-qwen3-14b-awq"
        assert MODEL_ALIAS_MAP["qwen3-14b-awq-baseline"] == "nram-qwen3-14b-awq"

    def test_unloaded_model_identities_are_not_mapped(self, engine):
        for model in (
            "nram-deepseek-r1-qwen-7b",
            "deepseek-r1-qwen-7b-baseline",
            "nram-gpt-oss-20b",
            "gpt-oss-20b-baseline",
        ):
            assert model not in MODEL_ALIAS_MAP
            with pytest.raises(SGLangEngineError):
                engine._resolve_upstream_model(make_request(model))

    def test_only_nram_alias_is_steering_enabled(self):
        assert "nram-qwen3-14b-awq" in NRAM_ENABLED_ALIASES
        assert "qwen3-14b-awq-baseline" not in NRAM_ENABLED_ALIASES

    def test_is_nram_enabled_true_for_nram_alias(self, engine):
        assert engine._is_nram_enabled(make_request("nram-qwen3-14b-awq"))

    def test_is_nram_enabled_false_for_baseline(self, engine):
        assert not engine._is_nram_enabled(make_request("qwen3-14b-awq-baseline"))

    def test_nram_disabled_via_option(self, engine):
        req = make_request("nram-qwen3-14b-awq", nram={"enabled": False})
        assert not engine._is_nram_enabled(req)


class TestSecurityBoundary:
    def test_forbidden_fields_constant(self):
        for field in [
            "custom_logit_processor",
            "custom_params",
            "serialized_processor",
            "processor_class",
            "python_code",
            "__req__",
        ]:
            assert field in _FORBIDDEN_CLIENT_FIELDS

    def test_client_cannot_inject_processor(self, engine):
        req = make_request(
            "nram-qwen3-14b-awq",
            nram={"custom_logit_processor": "import os; os.system('rm -rf /')"},
        )
        with pytest.raises(SGLangEngineError) as exc_info:
            engine._build_upstream_payload(req, nram_enabled=True)
        assert exc_info.value.status_code == 400

    def test_client_cannot_inject_serialized_processor(self, engine):
        req = make_request(
            "nram-qwen3-14b-awq",
            nram={"serialized_processor": "deadbeef"},
        )
        with pytest.raises(SGLangEngineError):
            engine._build_upstream_payload(req, nram_enabled=True)

    @pytest.mark.parametrize(
        "feature",
        [
            # NRAM v5 features are now runtime-wired and should be accepted
            # "dexperts",
            # "activation_addition",
            # "conceptor_steering",
            # "hidden_state_probes",
            # "latent_closed_loop",
            # "branch_tournament",
            # Only truly unsupported infrastructure features should be rejected
            "reft",
            "attention_head_gating",
            "kv_cache_firewall",
        ],
    )
    def test_unavailable_advanced_features_fail_loudly(self, engine, feature):
        req = make_request(
            "nram-qwen3-14b-awq",
            nram={"enabled": True, feature: True},
        )
        with pytest.raises(SGLangEngineError) as exc_info:
            engine._build_upstream_payload(req, nram_enabled=True)
        assert exc_info.value.status_code == 400
        assert feature in exc_info.value.message
    
    @pytest.mark.parametrize(
        "feature",
        [
            # NRAM v5 features are now runtime-wired and should be accepted
            "dexperts",
            "activation_addition",
            "conceptor_steering",
            "hidden_state_probes",
            "latent_closed_loop",
            "branch_tournament",
        ],
    )
    def test_nram_v5_features_are_now_wired(self, engine, feature):
        """NRAM v5 features are now runtime-wired and should be accepted."""
        req = make_request(
            "nram-qwen3-14b-awq",
            nram={"enabled": True, feature: True},
        )
        # Should NOT raise - feature is now wired
        try:
            payload = engine._build_upstream_payload(req, nram_enabled=True)
            # Verify the feature config is in the payload
            assert "custom_params" in payload
        except SGLangEngineError as e:
            if "Unsupported NRAM runtime feature" in str(e):
                pytest.fail(f"{feature} should be accepted, not rejected")
            raise


class TestPayloadConstruction:
    def test_baseline_has_no_processor(self, engine):
        req = make_request("qwen3-14b-awq-baseline")
        payload = engine._build_upstream_payload(req, nram_enabled=False)
        assert "custom_logit_processor" not in payload
        assert "custom_params" not in payload

    def test_nram_payload_has_processor_and_params(self, engine):
        req = make_request("nram-qwen3-14b-awq")
        payload = engine._build_upstream_payload(req, nram_enabled=True)
        assert "custom_logit_processor" in payload
        assert "custom_params" in payload
        params = payload["custom_params"]
        for key in [
            "positive_token_ids",
            "negative_token_ids",
            "forbidden_token_ids",
            "positive_bias",
            "negative_bias",
            "repetition_penalty",
            "profile",
        ]:
            assert key in params

    def test_user_content_preserved(self, engine):
        req = make_request("nram-qwen3-14b-awq")
        payload = engine._build_upstream_payload(req, nram_enabled=True)
        user_msgs = [m for m in payload["messages"] if m["role"] == "user"]
        assert user_msgs[0]["content"] == "Hello"

    def test_allowlist_filters_unknown_fields(self, engine):
        req = make_request("qwen3-14b-awq-baseline")
        # Inject an unknown attribute via dict (pydantic would reject, so test the allowlist directly)
        payload = engine._build_upstream_payload(req, nram_enabled=False)
        # Only allowlisted keys + model/messages should be present
        allowed = {
            "model", "messages", "temperature", "top_p", "max_tokens",
            "stream", "stop", "seed", "frequency_penalty", "presence_penalty",
            "response_format",
        }
        assert set(payload.keys()) <= allowed

    def test_model_alias_resolved_in_payload(self, engine):
        req = make_request("qwen3-14b-awq-baseline")
        payload = engine._build_upstream_payload(req, nram_enabled=False)
        assert payload["model"] == "nram-qwen3-14b-awq"

    def test_custom_params_profile_reflects_request(self, engine):
        req = make_request("nram-qwen3-14b-awq")
        payload = engine._build_upstream_payload(req, nram_enabled=True)
        assert payload["custom_params"]["profile"] == "visionary-psychedelic-keynote"

    def test_bias_values_within_bounds(self, engine):
        req = make_request("nram-qwen3-14b-awq")
        payload = engine._build_upstream_payload(req, nram_enabled=True)
        params = payload["custom_params"]
        assert 0.0 <= params["positive_bias"] <= 1.2
        assert 0.0 <= params["negative_bias"] <= 2.5
        assert 0.0 <= params["repetition_penalty"] <= 2.0
