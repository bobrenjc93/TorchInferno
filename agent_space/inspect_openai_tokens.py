from __future__ import annotations

import argparse

from torchinferno.cli import _build_openai_microbench_engine, _microbench_messages
from torchinferno.openai_server import _is_tensor_parallel_worker_model, _tensor_parallel_worker_loop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-mode", default="few-shot")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    engine_args = argparse.Namespace(
        backend="model",
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        model_kind="auto",
        tokenizer=None,
        tensor_parallel_size=8,
        devices=None,
        device=None,
        dtype="auto",
        max_model_len=None,
        trust_remote_code=True,
        token=None,
        revision=None,
        cache_dir=None,
        cache_backend="dense",
        page_size=16,
        llama_parallelism="tensor",
        max_batch_size=64,
        batch_wait_ms=10.0,
        synthetic_forward_sleep_us=0.0,
        vocab_size=32000,
    )
    engine = _build_openai_microbench_engine(engine_args)
    if _is_tensor_parallel_worker_model(engine.model):
        _tensor_parallel_worker_loop(engine)
        return
    try:
        messages = _microbench_messages(32, 0, args.prompt_mode)
        tokens = list(
            engine.generate_chat_tokens(
                messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        )
        print("stop_token_ids", sorted(engine.stop_token_ids))
        print("tokens", tokens)
        print("decoded", [engine.tokenizer.decode_token(token) for token in tokens])
    finally:
        engine.close()


if __name__ == "__main__":
    main()
