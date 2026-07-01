#!/usr/bin/env python3
"""Create a tiny random HF causal LM for fast offline GPU smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM


def build_tokenizer(vocab_size: int) -> PreTrainedTokenizerFast:
    base_tokens = [
        "<pad>",
        "<s>",
        "</s>",
        "<unk>",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "+",
        "-",
        "*",
        "/",
        "=",
        "?",
        ".",
        ",",
        ":",
        "Question",
        "Answer",
        "hello",
        "world",
    ]
    vocab = {token: idx for idx, token in enumerate(base_tokens)}
    for idx in range(len(vocab), vocab_size):
        vocab[f"tok{idx}"] = idx

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    )


def create_model(output_dir: Path, vocab_size: int) -> None:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=160,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        use_cache=True,
    )
    model = Qwen2ForCausalLM(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_tokenizer(vocab_size).save_pretrained(output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--vocab-size", type=int, default=128)
    args = parser.parse_args()

    if args.vocab_size < 32:
        raise ValueError("--vocab-size must be at least 32")
    create_model(args.output_dir, args.vocab_size)
    print(f"wrote tiny random Qwen2 model to {args.output_dir}")


if __name__ == "__main__":
    main()
