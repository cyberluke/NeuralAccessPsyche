"""Phase 12: reasoning-content stripping and stream sanitization tests."""
import json

import pytest

from core.engines.sglang_engine import strip_reasoning, _StreamState

OPEN = chr(60) + "think" + chr(62)
CLOSE = chr(60) + "/think" + chr(62)


class TestStripReasoning:
    def test_removes_complete_think_block(self):
        text = OPEN + "Let me think.\n" + CLOSE + "\nThe answer is 4."
        assert strip_reasoning(text) == "The answer is 4."

    def test_no_think_block_unchanged(self):
        text = "Just a normal answer."
        assert strip_reasoning(text) == "Just a normal answer."

    def test_multiple_blocks_removed(self):
        text = OPEN + "a\n" + CLOSE + "\n" + OPEN + "b\n" + CLOSE + "\nc"
        assert strip_reasoning(text) == "c"

    def test_empty_after_strip(self):
        assert strip_reasoning(OPEN + "only reasoning" + CLOSE) == ""

    def test_multiline_reasoning(self):
        text = OPEN + "\nstep 1\nstep 2\nstep 3\n" + CLOSE + "\nFinal."
        assert strip_reasoning(text) == "Final."


class TestStreamState:
    def _chunk(self, content=None, finish_reason=None, model="upstream-model"):
        delta = {}
        if content is not None:
            delta["content"] = content
        choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
        return "data: " + json.dumps(
            {"id": "x", "object": "chat.completion.chunk", "model": model, "choices": [choice]}
        )

    def test_plain_content_passes_through(self):
        state = _StreamState("public-alias")
        out = state.feed(self._chunk(content="Hello"))
        assert len(out) == 1
        parsed = json.loads(out[0][len("data: "):])
        assert parsed["choices"][0]["delta"]["content"] == "Hello"
        assert parsed["model"] == "public-alias"

    def test_model_rewritten_to_public_alias(self):
        state = _StreamState("nram-deepseek-r1-qwen-7b")
        out = state.feed(self._chunk(content="x", model="internal/secret-model"))
        parsed = json.loads(out[0][len("data: "):])
        assert parsed["model"] == "nram-deepseek-r1-qwen-7b"

    def test_complete_reasoning_block_suppressed(self):
        state = _StreamState("alias")
        out = state.feed(self._chunk(content=OPEN + "reasoning\n" + CLOSE + "\nAnswer"))
        # Only the visible "Answer" should appear.
        combined = "".join(
            json.loads(o[len("data: "):])["choices"][0]["delta"].get("content", "")
            for o in out
        )
        assert "reasoning" not in combined
        assert "Answer" in combined

    def test_partial_reasoning_buffered_until_close(self):
        state = _StreamState("alias")
        # Reasoning starts but does not close in this chunk.
        out1 = state.feed(self._chunk(content="visible1\n" + OPEN + "\nhidden..."))
        # visible1 emitted, hidden buffered
        text1 = "".join(
            json.loads(o[len("data: "):])["choices"][0]["delta"].get("content", "")
            for o in out1
        )
        assert "visible1" in text1
        assert "hidden" not in text1

        out2 = state.feed(self._chunk(content="more hidden\n" + CLOSE + "\nvisible2"))
        text2 = "".join(
            json.loads(o[len("data: "):])["choices"][0]["delta"].get("content", "")
            for o in out2
        )
        assert "hidden" not in text2
        assert "more hidden" not in text2
        assert "visible2" in text2

    def test_done_line_yields_nothing(self):
        state = _StreamState("alias")
        assert state.feed("data: [DONE]") == []

    def test_unparseable_line_dropped(self):
        state = _StreamState("alias")
        assert state.feed("data: {not valid json") == []

    def test_finish_reason_chunk_preserved(self):
        state = _StreamState("alias")
        out = state.feed(self._chunk(content=None, finish_reason="stop"))
        assert len(out) == 1
        parsed = json.loads(out[0][len("data: "):])
        assert parsed["choices"][0]["finish_reason"] == "stop"
