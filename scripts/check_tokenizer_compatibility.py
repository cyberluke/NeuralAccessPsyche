#!/usr/bin/env python3
"""
Tokenizer compatibility gate for DExperts.

Compares Qwen3-14B-AWQ vs Qwen3-0.6B-Base tokenizers to ensure
LoRA adapters trained on 0.6B can be applied to 14B model.

Checks:
- vocab_size equality
- token<->id mapping equality
- BOS/EOS/PAD IDs
- Special tokens
- Normalizer and pre-tokenizer
- Test corpus encoding equality
- Accept only if exact_vocab_mapping_equal = true
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

from transformers import AutoTokenizer


def load_tokenizer(model_path: str, revision: str = None) -> AutoTokenizer:
    """Load tokenizer from model path."""
    print(f"Loading tokenizer from {model_path}")
    if revision:
        tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def compare_vocab_sizes(tok1: AutoTokenizer, tok2: AutoTokenizer) -> Dict[str, Any]:
    """Compare vocabulary sizes."""
    vocab1 = tok1.vocab_size
    vocab2 = tok2.vocab_size
    return {
        "vocab_size_1": vocab1,
        "vocab_size_2": vocab2,
        "equal": vocab1 == vocab2,
    }


def compare_special_tokens(tok1: AutoTokenizer, tok2: AutoTokenizer) -> Dict[str, Any]:
    """Compare special token IDs."""
    result = {}

    for attr in ["bos_token_id", "eos_token_id", "pad_token_id"]:
        id1 = getattr(tok1, attr, None)
        id2 = getattr(tok2, attr, None)
        result[attr] = {
            "model_1": id1,
            "model_2": id2,
            "equal": id1 == id2,
        }

    # Compare special tokens dict
    special1 = tok1.special_tokens_map
    special2 = tok2.special_tokens_map

    result["special_tokens_equal"] = special1 == special2
    result["special_tokens_1"] = special1
    result["special_tokens_2"] = special2

    return result


def compare_token_mappings(tok1: AutoTokenizer, tok2: AutoTokenizer, sample_size: int = 1000) -> Dict[str, Any]:
    """Compare token<->id mappings for a sample of tokens."""
    vocab1 = tok1.get_vocab()
    vocab2 = tok2.get_vocab()

    keys1 = set(vocab1.keys())
    keys2 = set(vocab2.keys())

    common_keys = keys1 & keys2
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1

    sample_keys = sorted(list(common_keys))[:sample_size]

    mapping_equal = True
    mismatches = []

    for token in sample_keys:
        id1 = vocab1[token]
        id2 = vocab2[token]
        if id1 != id2:
            mapping_equal = False
            mismatches.append({
                "token": token,
                "id_1": id1,
                "id_2": id2,
            })
            if len(mismatches) >= 10:
                break

    return {
        "vocab_keys_equal": keys1 == keys2,
        "common_tokens": len(common_keys),
        "only_in_model_1": len(only_in_1),
        "only_in_model_2": len(only_in_2),
        "sample_size": len(sample_keys),
        "mappings_equal": mapping_equal,
        "mismatches": mismatches[:10],
    }


def compare_tokenizer_configs(tok1: AutoTokenizer, tok2: AutoTokenizer) -> Dict[str, Any]:
    """Compare tokenizer configuration."""
    result = {}

    if hasattr(tok1, "backend_tokenizer") and hasattr(tok2, "backend_tokenizer"):
        backend1 = tok1.backend_tokenizer
        backend2 = tok2.backend_tokenizer

        norm1 = str(backend1.normalizer) if backend1.normalizer else None
        norm2 = str(backend2.normalizer) if backend2.normalizer else None
        result["normalizer_equal"] = norm1 == norm2
        result["normalizer_1"] = norm1
        result["normalizer_2"] = norm2

        pretok1 = str(backend1.pre_tokenizer) if backend1.pre_tokenizer else None
        pretok2 = str(backend2.pre_tokenizer) if backend2.pre_tokenizer else None
        result["pre_tokenizer_equal"] = pretok1 == pretok2
        result["pre_tokenizer_1"] = pretok1
        result["pre_tokenizer_2"] = pretok2

        model1 = str(backend1.model) if backend1.model else None
        model2 = str(backend2.model) if backend2.model else None
        result["model_equal"] = model1 == model2
        result["model_1"] = model1
        result["model_2"] = model2

    return result


def compare_encoding(tok1: AutoTokenizer, tok2: AutoTokenizer, test_corpus: List[str]) -> Dict[str, Any]:
    """Compare encoding of test corpus."""
    results = []
    all_equal = True

    for text in test_corpus:
        enc1 = tok1.encode(text)
        enc2 = tok2.encode(text)

        equal = enc1 == enc2
        if not equal:
            all_equal = False

        results.append({
            "text": text[:50] + "..." if len(text) > 50 else text,
            "length_1": len(enc1),
            "length_2": len(enc2),
            "equal": equal,
        })

    return {
        "all_equal": all_equal,
        "corpus_size": len(test_corpus),
        "results": results,
    }


def main():
    """Main entry point."""
    model_14b = r"E:\_MODELS\huggingface\hub\models--Qwen--Qwen3-14B-AWQ\snapshots\31c69efc29464b6bb0aee1398b5a7b50a99340c3"
    model_06b = r"E:\_MODELS\huggingface\hub\models--Qwen--Qwen3-0.6B-Base\snapshots\da87bfb608c14b7cf20ba1ce41287e8de496c0cd"

    if not Path(model_14b).exists():
        print(f"ERROR: Model path does not exist: {model_14b}")
        sys.exit(1)

    if not Path(model_06b).exists():
        print(f"ERROR: Model path does not exist: {model_06b}")
        sys.exit(1)

    print("=" * 80)
    print("TOKENIZER COMPATIBILITY GATE")
    print("=" * 80)
    print()

    tok_14b = load_tokenizer(model_14b)
    tok_06b = load_tokenizer(model_06b)

    test_corpus = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "NRAM v5 implements DExperts for inference-time steering.",
        "Toxicity detection is important for safe AI systems.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "1234567890 !@#$%^&*()",
        "Multiple   spaces   and\nnewlines\n\nhere.",
        "Unicode: \u65e5\u672c\u8a9e\u30c6\u30b9\u30c8 \U0001f389\U0001f680",
    ]

    print("\n[1/6] Comparing vocabulary sizes...")
    vocab_comparison = compare_vocab_sizes(tok_14b, tok_06b)
    print(f"  14B vocab: {vocab_comparison['vocab_size_1']}")
    print(f"  0.6B vocab: {vocab_comparison['vocab_size_2']}")
    print(f"  Equal: {vocab_comparison['equal']}")

    print("\n[2/6] Comparing special tokens...")
    special_comparison = compare_special_tokens(tok_14b, tok_06b)
    print(f"  BOS equal: {special_comparison['bos_token_id']['equal']}")
    print(f"  EOS equal: {special_comparison['eos_token_id']['equal']}")
    print(f"  PAD equal: {special_comparison['pad_token_id']['equal']}")

    print("\n[3/6] Comparing token mappings...")
    mapping_comparison = compare_token_mappings(tok_14b, tok_06b, sample_size=1000)
    print(f"  Common tokens: {mapping_comparison['common_tokens']}")
    print(f"  Only in 14B: {mapping_comparison['only_in_model_1']}")
    print(f"  Only in 0.6B: {mapping_comparison['only_in_model_2']}")
    print(f"  Mappings equal: {mapping_comparison['mappings_equal']}")
    if mapping_comparison['mismatches']:
        print(f"  First mismatch: {mapping_comparison['mismatches'][0]}")

    print("\n[4/6] Comparing tokenizer configurations...")
    config_comparison = compare_tokenizer_configs(tok_14b, tok_06b)
    print(f"  Normalizer equal: {config_comparison.get('normalizer_equal', 'N/A')}")
    print(f"  Pre-tokenizer equal: {config_comparison.get('pre_tokenizer_equal', 'N/A')}")
    print(f"  Model equal: {config_comparison.get('model_equal', 'N/A')}")

    print("\n[5/6] Comparing test corpus encoding...")
    encoding_comparison = compare_encoding(tok_14b, tok_06b, test_corpus)
    print(f"  All encodings equal: {encoding_comparison['all_equal']}")
    print(f"  Corpus size: {encoding_comparison['corpus_size']}")

    # Final verdict
    print("\n[6/6] Computing final verdict...")

    # EOS token ID difference is a known Qwen3 configuration difference
    # between base (