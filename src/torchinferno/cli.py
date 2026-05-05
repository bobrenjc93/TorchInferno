from __future__ import annotations

import argparse
import time

import torch

from torchinferno.models.dsv4 import DSv4ForCausalLM, tiny_dsv4_config
from torchinferno.runtime.batching import InferenceRequest, run_continuous_batch


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_dsv4_smoke(args: argparse.Namespace) -> int:
    device = torch.device(args.device) if args.device else _default_device()
    torch.manual_seed(args.seed)
    config = tiny_dsv4_config(
        vocab_size=args.vocab_size,
        max_seq_len=args.prompt_tokens + args.new_tokens + 8,
    )
    model = DSv4ForCausalLM(config).to(device).eval()
    if args.compile:
        model.forward = torch.compile(model.forward, mode="reduce-overhead")  # type: ignore[method-assign]

    prompt = torch.randint(0, config.vocab_size, (args.batch_size, args.prompt_tokens), device=device)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(prompt, max_new_tokens=args.new_tokens, temperature=args.temperature)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print("TorchInferno DSv4 smoke")
    print(f"device={device} batch={args.batch_size} prompt={args.prompt_tokens} new={args.new_tokens}")
    print(f"shape={tuple(output.shape)} elapsed_ms={elapsed_ms:.2f}")
    print(f"tokens[0]={output[0].tolist()}")
    return 0


def run_batch_smoke(args: argparse.Namespace) -> int:
    device = torch.device(args.device) if args.device else _default_device()
    torch.manual_seed(args.seed)
    config = tiny_dsv4_config(max_seq_len=64)
    model = DSv4ForCausalLM(config).to(device).eval()
    requests = [
        InferenceRequest("req-a", (1, 2, 3), 2),
        InferenceRequest("req-b", (4, 5), 3),
        InferenceRequest("req-c", (6, 7, 8), 1),
    ]
    with torch.inference_mode():
        results = run_continuous_batch(model, requests, device=device)
    for result in results:
        print(f"{result.request_id}: {list(result.tokens)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="torchinferno")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("dsv4-smoke", help="Run a DSv4 end-to-end generation smoke test.")
    smoke.add_argument("--device", default=None, help="Torch device, defaults to cuda when available.")
    smoke.add_argument("--seed", type=int, default=0)
    smoke.add_argument("--batch-size", type=int, default=2)
    smoke.add_argument("--prompt-tokens", type=int, default=8)
    smoke.add_argument("--new-tokens", type=int, default=4)
    smoke.add_argument("--vocab-size", type=int, default=128)
    smoke.add_argument("--temperature", type=float, default=0.0)
    smoke.add_argument("--compile", action="store_true", help="Compile the DSv4 forward path with torch.compile.")
    smoke.set_defaults(func=run_dsv4_smoke)

    batch = subparsers.add_parser("batch-smoke", help="Run the ragged request batching harness.")
    batch.add_argument("--device", default=None, help="Torch device, defaults to cuda when available.")
    batch.add_argument("--seed", type=int, default=0)
    batch.set_defaults(func=run_batch_smoke)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
