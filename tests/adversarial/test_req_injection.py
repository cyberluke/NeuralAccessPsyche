"""Security tests for SGLang's trusted, server-side ``__req__`` injection.

The application sends only JSON ``custom_params``.  SGLang 0.5.16 creates a
shallow copy in ``Req.__init__`` and adds its own scheduler ``Req`` object
in-process.  An API request object must never cross the HTTP boundary.
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from core.contracts.openai import ChatCompletionRequest
from core.engines.sglang_engine import SGLangEngine
from core.steering.serialization import build_custom_params
from nram_sglang.processor import NRAMLogitProcessor


class TestReqInjection:
    """Falsify unsafe client injection while preserving the official hook."""

    def test_build_custom_params_excludes_application_request(self):
        application_request = SimpleNamespace(output_ids=[10, 20, 30])

        custom_params = build_custom_params(
            positive_token_ids=[1, 2, 3],
            negative_token_ids=[4, 5, 6],
            forbidden_token_ids=[7, 8, 9],
            positive_bias=0.5,
            negative_bias=0.3,
            repetition_penalty=1.2,
            profile="peak",
            max_tokens=512,
            phenomenon_weights={"overlap": 0.8, "forgetting": 0.6},
            request=application_request,
        )

        assert "__req__" not in custom_params
        assert "request" not in custom_params
        # This is the actual wire contract; non-JSON request objects would fail.
        assert json.loads(json.dumps(custom_params)) == custom_params

    def test_sglang_payload_keeps_request_object_off_wire(self):
        tokenizer = MagicMock()
        tokenizer.get_vocab.return_value = {"token1": 1, "token2": 2, "token3": 3}
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.return_value = "test"
        engine = SGLangEngine(
            base_url="http://localhost:30000/v1",
            model="nram-qwen3-14b-awq",
            tokenizer=tokenizer,
        )
        request = ChatCompletionRequest(
            model="nram-qwen3-14b-awq",
            messages=[{"role": "user", "content": "Test prompt"}],
            nram={
                "enabled": True,
                "profile": "peak",
                "phenomenon_weights": {"overlap": 0.8},
            },
            max_tokens=32,
        )

        payload = engine._build_upstream_payload(request, nram_enabled=True)

        assert "custom_logit_processor" in payload
        assert "__req__" not in payload["custom_params"]
        # Exact payload construction must remain JSON serializable.
        json.dumps(payload)

    def test_processor_consumes_trusted_server_request_history(self):
        """Model the documented SGLang ``Req.__init__`` in-process merge."""
        wire_params = build_custom_params(
            positive_token_ids=[],
            negative_token_ids=[],
            forbidden_token_ids=[],
            positive_bias=0.0,
            negative_bias=0.0,
            repetition_penalty=0.8,
            profile="peak",
            max_tokens=32,
            phenomenon_weights={"overlap": 0.8},
        )
        server_params = wire_params | {
            "__req__": SimpleNamespace(origin_input_ids=[1, 2], output_ids=[10, 20, 30])
        }
        logits = np.zeros((1, 100), dtype=np.float32)

        result = NRAMLogitProcessor()(logits, [server_params])

        # Repetition (-0.8) plus overlap (+0.28) leaves a direct history effect.
        assert np.isclose(result[0, 10], -0.52)
        assert result[0, 60] == 0.0
