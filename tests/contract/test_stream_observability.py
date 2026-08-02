import asyncio
import json

from api.routes import ChatCompletionRequest, _sse_observability
from core.contracts.observability import NRAMStreamEvent, ObservabilityLevel


async def _source(*chunks):
    for chunk in chunks:
        yield chunk


def _events(payload):
    return [
        NRAMStreamEvent.model_validate(json.loads(line[6:]))
        for chunk in payload
        if b"event: nram" in chunk
        for line in chunk.decode().splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def test_minimal_mode_preserves_normal_openai_stream_without_telemetry():
    request = ChatCompletionRequest(
        model="nram-qwen3-14b-awq", messages=[{"role": "user", "content": "x"}],
        stream=True, nram={"observability_level": "minimal"},
    )
    source = _source(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n', b"data: [DONE]\n\n")
    result = asyncio.run(_collect(_sse_observability(source, request)))
    assert not any(b"event: nram" in item for item in result)


def test_research_events_are_ordered_and_request_scoped():
    request = ChatCompletionRequest(
        model="nram-qwen3-14b-awq", messages=[{"role": "user", "content": "x"}],
        stream=True, nram={"observability_level": "research", "request_id": "req-a"},
    )
    source = _source(b'data: {"choices":[{"delta":{"content":"A"}}]}\n\n', b"data: [DONE]\n\n")
    result = asyncio.run(_collect(_sse_observability(source, request)))
    events = _events(result)
    assert [event.sequence_number for event in events] == list(range(len(events)))
    assert {event.request_id for event in events} == {"req-a"}
    token = next(event for event in events if event.event_type.value == "token")
    assert token.payload["token_id"] is None
    assert token.payload["availability_reason"]
    assert "logits" not in json.dumps(token.payload).lower()


async def _collect(iterator):
    return [item async for item in iterator]
