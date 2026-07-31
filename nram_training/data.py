"""Qwen3 formatting and assistant-only label contract."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Example:
    user: str
    assistant: str

def assistant_mask(input_ids, assistant_start: int):
    labels = input_ids.clone() if hasattr(input_ids, "clone") else list(input_ids)
    labels[:assistant_start] = -100
    return labels

def render_qwen3(tokenizer, example: Example) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": example.user}, {"role": "assistant", "content": example.assistant}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False)
