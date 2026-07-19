#!/usr/bin/env python3
"""Compare mixed decode/prefill paths with isolated reference steps."""

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist

from torchinferno.models.llama3.tensor_parallel import Llama3TensorParallelForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--decode-steps", type=int, default=8)
    args = parser.parse_args()

    model = Llama3TensorParallelForCausalLM.from_pretrained(
        args.checkpoint,
        dtype="bfloat16",
    ).eval()
    device = model.device
    rank = dist.get_rank() if dist.is_initialized() else 0
    generator = torch.Generator(device=device).manual_seed(17)
    prompt_a = torch.randint(
        0,
        model.config.vocab_size,
        (1, 32),
        generator=generator,
        device=device,
    )
    prompt_b = torch.randint(
        0,
        model.config.vocab_size,
        (1, 48),
        generator=generator,
        device=device,
    )
    cache_mixed = model.allocate_cache(2, 128)
    cache_separate = model.allocate_cache(2, 128)

    with torch.inference_mode():
        logits_a_mixed, _ = model.forward(
            prompt_a,
            cache=cache_mixed.for_rows((0,)),
            return_last_logits_only=True,
            return_sharded_logits=True,
        )
        model.forward(
            prompt_a,
            cache=cache_separate.for_rows((0,)),
            return_last_logits_only=True,
            return_sharded_logits=True,
        )
        token_a = torch.argmax(model._gather_logits(logits_a_mixed)[:, -1, :], dim=-1)

        mixed_ids = torch.zeros((2, 48), dtype=torch.long, device=device)
        mixed_ids[0, 0] = token_a[0]
        mixed_ids[1] = prompt_b[0]
        mixed_positions = torch.empty((2, 48), dtype=torch.long, device=device)
        mixed_positions[0] = torch.arange(32, 80, device=device)
        mixed_positions[1] = torch.arange(48, device=device)
        mixed_logits = model.forward_step_flashinfer(
            mixed_ids,
            cache_mixed,
            seq_lens=torch.tensor([32, 0], dtype=torch.long, device=device),
            q_lens=torch.tensor([1, 48], dtype=torch.long, device=device),
            write_positions=mixed_positions,
            logit_positions=torch.tensor([0, 47], dtype=torch.long, device=device),
            row_indices=torch.tensor([0, 1], dtype=torch.long, device=device),
        )

        decode_logits = model.forward_step_flashinfer(
            token_a.view(1, 1),
            cache_separate,
            seq_lens=torch.tensor([32], dtype=torch.long, device=device),
            q_lens=torch.tensor([1], dtype=torch.long, device=device),
            write_positions=torch.tensor([[32]], dtype=torch.long, device=device),
            logit_positions=torch.tensor([0], dtype=torch.long, device=device),
            row_indices=torch.tensor([0], dtype=torch.long, device=device),
        )
        prefill_logits = model.forward_step_flashinfer(
            prompt_b,
            cache_separate,
            seq_lens=torch.tensor([0], dtype=torch.long, device=device),
            q_lens=torch.tensor([48], dtype=torch.long, device=device),
            write_positions=torch.arange(48, device=device).view(1, 48),
            logit_positions=torch.tensor([47], dtype=torch.long, device=device),
            row_indices=torch.tensor([1], dtype=torch.long, device=device),
        )

    expected_logits = torch.cat((decode_logits, prefill_logits), dim=0)
    logit_error = float((mixed_logits - expected_logits).abs().max().item())
    layer_errors = []
    for mixed_layer, separate_layer in zip(cache_mixed.layers, cache_separate.layers):
        row_errors = []
        for row, valid_tokens in ((0, 33), (1, 48)):
            key_error = (
                mixed_layer.keys[row, :, :valid_tokens]
                - separate_layer.keys[row, :, :valid_tokens]
            ).abs().max()
            value_error = (
                mixed_layer.values[row, :, :valid_tokens]
                - separate_layer.values[row, :, :valid_tokens]
            ).abs().max()
            row_errors.append(max(float(key_error.item()), float(value_error.item())))
        layer_errors.append(max(row_errors))
    predicted_mixed = torch.argmax(model._gather_logits(mixed_logits), dim=-1).tolist()
    predicted_separate = torch.argmax(model._gather_logits(expected_logits), dim=-1).tolist()
    print(
        {
            "case": "eager_first_step",
            "rank": rank,
            "max_logit_error": logit_error,
            "max_kv_error": max(layer_errors),
            "first_bad_layer": next(
                (index for index, error in enumerate(layer_errors) if error != 0.0),
                None,
            ),
            "mixed_tokens": predicted_mixed,
            "separate_tokens": predicted_separate,
        },
        flush=True,
    )

    # Reproduce the fixed-capacity serving graph exactly: two live requests,
    # zero-query padding to 64 rows, and token padding assigned after the final
    # live request. Compare it with isolated row calls through subsequent
    # greedy decode steps so latent KV corruption cannot hide behind first-token
    # agreement.
    graph_cache = model.allocate_cache(64, 128)
    graph_reference_cache = model.allocate_cache(64, 128)
    with torch.inference_mode():
        graph_prompt_logits, _ = model.forward(
            prompt_a,
            cache=graph_cache.for_rows((0,)),
            return_last_logits_only=True,
            return_sharded_logits=True,
        )
        model.forward(
            prompt_a,
            cache=graph_reference_cache.for_rows((0,)),
            return_last_logits_only=True,
            return_sharded_logits=True,
        )
        graph_token_a = torch.argmax(
            model._gather_logits(graph_prompt_logits)[:, -1, :],
            dim=-1,
        )

        token_capacity = 128
        graph_input_ids = torch.zeros(
            (1, token_capacity),
            dtype=torch.long,
            device=device,
        )
        graph_input_ids[0, 0] = graph_token_a[0]
        graph_input_ids[0, 1:49] = prompt_b[0]
        graph_start_positions = torch.tensor(
            [32, 0, *([0] * 62)],
            dtype=torch.long,
            device=device,
        )
        graph_q_lens = torch.tensor(
            [1, 48, *([1] * 61), 18],
            dtype=torch.long,
            device=device,
        )
        graph_row_indices = torch.tensor(
            list(range(64)),
            dtype=torch.long,
            device=device,
        )
        graph_write_positions = torch.cat(
            (
                torch.tensor([32], dtype=torch.long, device=device),
                torch.arange(48, dtype=torch.long, device=device),
                torch.zeros(61, dtype=torch.long, device=device),
                torch.arange(18, dtype=torch.long, device=device),
            )
        )
        graph_flat_rows = torch.tensor(
            [0, *([1] * 48), *range(2, 63), *([63] * 18)],
            dtype=torch.long,
            device=device,
        )
        graph_logit_positions = torch.tensor(
            [0, 48, *range(49, 110), 110],
            dtype=torch.long,
            device=device,
        )
        graph_logits = model.try_prefill_token_bucket_fa3_graph(
            graph_input_ids,
            graph_cache,
            start_positions=graph_start_positions,
            q_lens=graph_q_lens,
            row_indices=graph_row_indices,
            write_positions=graph_write_positions,
            flat_rows=graph_flat_rows,
            logit_positions=graph_logit_positions,
        )
        if graph_logits is None:
            raise RuntimeError("token-bucket FA3 graph was unavailable")

        reference_decode_logits = model.forward_step_flashinfer(
            graph_token_a.view(1, 1),
            graph_reference_cache,
            seq_lens=torch.tensor([32], dtype=torch.long, device=device),
            q_lens=torch.tensor([1], dtype=torch.long, device=device),
            write_positions=torch.tensor([[32]], dtype=torch.long, device=device),
            logit_positions=torch.tensor([0], dtype=torch.long, device=device),
            row_indices=torch.tensor([0], dtype=torch.long, device=device),
        )
        reference_prefill_logits = model.forward_step_flashinfer(
            prompt_b,
            graph_reference_cache,
            seq_lens=torch.tensor([0], dtype=torch.long, device=device),
            q_lens=torch.tensor([48], dtype=torch.long, device=device),
            write_positions=torch.arange(48, device=device).view(1, 48),
            logit_positions=torch.tensor([47], dtype=torch.long, device=device),
            row_indices=torch.tensor([1], dtype=torch.long, device=device),
        )
        reference_logits = torch.cat(
            (reference_decode_logits, reference_prefill_logits),
            dim=0,
        )
        graph_tokens = torch.argmax(
            model._gather_logits(graph_logits[:2, -1, :]),
            dim=-1,
        )
        reference_tokens = torch.argmax(
            model._gather_logits(reference_logits[:, -1, :]),
            dim=-1,
        )
        first_graph_tokens = graph_tokens.tolist()
        first_reference_tokens = reference_tokens.tolist()
        graph_trajectories = [[int(token)] for token in first_graph_tokens]
        reference_trajectories = [[int(token)] for token in first_reference_tokens]
        graph_seq_lens = [33, 48]
        reference_seq_lens = [33, 48]

        for _ in range(max(0, args.decode_steps)):
            graph_decode_input = graph_tokens.view(2, 1)
            graph_decode_logits = model.forward_step_flashinfer(
                graph_decode_input,
                graph_cache,
                seq_lens=torch.tensor(graph_seq_lens, dtype=torch.long, device=device),
                q_lens=torch.ones(2, dtype=torch.long, device=device),
                write_positions=torch.tensor(
                    graph_seq_lens,
                    dtype=torch.long,
                    device=device,
                ).view(2, 1),
                logit_positions=torch.zeros(2, dtype=torch.long, device=device),
                row_indices=torch.tensor([0, 1], dtype=torch.long, device=device),
            )
            next_graph_tokens = torch.argmax(
                model._gather_logits(graph_decode_logits[:, -1, :]),
                dim=-1,
            )
            next_reference_tokens = []
            for row in range(2):
                row_logits = model.forward_step_flashinfer(
                    reference_tokens[row].view(1, 1),
                    graph_reference_cache,
                    seq_lens=torch.tensor(
                        [reference_seq_lens[row]],
                        dtype=torch.long,
                        device=device,
                    ),
                    q_lens=torch.ones(1, dtype=torch.long, device=device),
                    write_positions=torch.tensor(
                        [[reference_seq_lens[row]]],
                        dtype=torch.long,
                        device=device,
                    ),
                    logit_positions=torch.zeros(1, dtype=torch.long, device=device),
                    row_indices=torch.tensor([row], dtype=torch.long, device=device),
                )
                next_reference_tokens.append(
                    torch.argmax(
                        model._gather_logits(row_logits[:, -1, :]),
                        dim=-1,
                    )[0]
                )
            reference_tokens = torch.stack(next_reference_tokens)
            graph_tokens = next_graph_tokens
            for row in range(2):
                graph_trajectories[row].append(int(graph_tokens[row].item()))
                reference_trajectories[row].append(int(reference_tokens[row].item()))
                graph_seq_lens[row] += 1
                reference_seq_lens[row] += 1

    graph_layer_errors = []
    for graph_layer, reference_layer in zip(
        graph_cache.layers,
        graph_reference_cache.layers,
    ):
        row_errors = []
        for row, valid_tokens in enumerate(graph_seq_lens):
            key_error = (
                graph_layer.keys[row, :, :valid_tokens]
                - reference_layer.keys[row, :, :valid_tokens]
            ).abs().max()
            value_error = (
                graph_layer.values[row, :, :valid_tokens]
                - reference_layer.values[row, :, :valid_tokens]
            ).abs().max()
            row_errors.append(max(float(key_error.item()), float(value_error.item())))
        graph_layer_errors.append(max(row_errors))
    print(
        {
            "case": "token_graph_then_decode",
            "rank": rank,
            "first_max_logit_error": float(
                (graph_logits[:2] - reference_logits).abs().max().item()
            ),
            "max_kv_error": max(graph_layer_errors),
            "first_bad_layer": next(
                (index for index, error in enumerate(graph_layer_errors) if error != 0.0),
                None,
            ),
            "graph_tokens": graph_trajectories,
            "reference_tokens": reference_trajectories,
            "trajectory_match": graph_trajectories == reference_trajectories,
        },
        flush=True,
    )

    prefix_graph_cache = model.allocate_cache(65, 192)
    prefix_reference_cache = model.allocate_cache(65, 192)
    prefix_row = 64
    with torch.inference_mode():
        model.forward(
            prompt_a,
            cache=prefix_graph_cache.for_rows((prefix_row,)),
            return_last_logits_only=True,
            return_sharded_logits=True,
        )
        model.forward(
            prompt_a,
            cache=prefix_reference_cache.for_rows((prefix_row,)),
            return_last_logits_only=True,
            return_sharded_logits=True,
        )

    def run_prefix_replay_case(suffixes: list[torch.Tensor]) -> dict[str, object]:
        real_batch = len(suffixes)
        pad_count = 64 - real_batch
        sequences = [suffix.tolist() for suffix in suffixes] + [[0] for _ in range(pad_count)]
        real_q_lens = [len(sequence) for sequence in sequences]
        starts = [prompt_a.size(1)] * real_batch + [0] * pad_count
        rows = list(range(64))
        sources = [prefix_row] * real_batch + list(range(real_batch, 64))
        copy_lens = [prompt_a.size(1)] * real_batch + [0] * pad_count
        token_capacity = 128
        bucket_q_lens = list(real_q_lens)
        bucket_q_lens[-1] += token_capacity - sum(bucket_q_lens)

        packed_tokens: list[int] = []
        write_positions: list[int] = []
        flat_rows: list[int] = []
        logit_positions: list[int] = []
        for sequence, real_q_len, bucket_q_len, start, row in zip(
            sequences,
            real_q_lens,
            bucket_q_lens,
            starts,
            rows,
        ):
            logit_positions.append(len(packed_tokens) + real_q_len - 1)
            packed_tokens.extend(int(token) for token in sequence)
            packed_tokens.extend([0] * (bucket_q_len - real_q_len))
            write_positions.extend(range(start, start + bucket_q_len))
            flat_rows.extend([row] * bucket_q_len)

        starts_t = torch.tensor(starts, dtype=torch.long, device=device)
        rows_t = torch.tensor(rows, dtype=torch.long, device=device)
        sources_t = torch.tensor(sources, dtype=torch.long, device=device)
        copy_lens_t = torch.tensor(copy_lens, dtype=torch.long, device=device)
        copied = model.try_copy_token_bucket_prefix_graph(
            prefix_graph_cache,
            start_positions=copy_lens_t,
            row_indices=rows_t,
            src_prefix_rows=sources_t,
            prefix_copy_capacity=64,
        )
        if not copied:
            raise RuntimeError("token-bucket prefix graph was unavailable")
        graph_logits = model.try_prefill_token_bucket_fa3_graph(
            torch.tensor([packed_tokens], dtype=torch.long, device=device),
            prefix_graph_cache,
            start_positions=starts_t,
            q_lens=torch.tensor(bucket_q_lens, dtype=torch.long, device=device),
            row_indices=rows_t,
            write_positions=torch.tensor(write_positions, dtype=torch.long, device=device),
            flat_rows=torch.tensor(flat_rows, dtype=torch.long, device=device),
            logit_positions=torch.tensor(logit_positions, dtype=torch.long, device=device),
        )
        if graph_logits is None:
            raise RuntimeError("token-bucket prefix replay graph was unavailable")

        reference_logits = []
        for row, suffix in enumerate(suffixes):
            prefix_reference_cache.copy_prefix_from(
                prefix_reference_cache,
                prompt_a.size(1),
                source_row=prefix_row,
                dest_row=row,
            )
            row_logits = model.forward_step_flashinfer(
                suffix.view(1, -1),
                prefix_reference_cache,
                seq_lens=torch.tensor(
                    [prompt_a.size(1)],
                    dtype=torch.long,
                    device=device,
                ),
                q_lens=torch.tensor([suffix.numel()], dtype=torch.long, device=device),
                write_positions=torch.arange(
                    prompt_a.size(1),
                    prompt_a.size(1) + suffix.numel(),
                    dtype=torch.long,
                    device=device,
                ).view(1, -1),
                logit_positions=torch.tensor(
                    [suffix.numel() - 1],
                    dtype=torch.long,
                    device=device,
                ),
                row_indices=torch.tensor([row], dtype=torch.long, device=device),
            )
            reference_logits.append(row_logits)
        reference = torch.cat(reference_logits, dim=0)
        graph_live = graph_logits[:real_batch]
        graph_predictions = torch.argmax(model._gather_logits(graph_live), dim=-1).tolist()
        reference_predictions = torch.argmax(model._gather_logits(reference), dim=-1).tolist()
        source_errors = []
        live_errors = []
        for graph_layer, reference_layer in zip(
            prefix_graph_cache.layers,
            prefix_reference_cache.layers,
        ):
            source_errors.append(
                max(
                    float(
                        (
                            graph_layer.keys[prefix_row, :, : prompt_a.size(1)]
                            - reference_layer.keys[prefix_row, :, : prompt_a.size(1)]
                        ).abs().max().item()
                    ),
                    float(
                        (
                            graph_layer.values[prefix_row, :, : prompt_a.size(1)]
                            - reference_layer.values[prefix_row, :, : prompt_a.size(1)]
                        ).abs().max().item()
                    ),
                )
            )
            for row, suffix in enumerate(suffixes):
                valid = prompt_a.size(1) + suffix.numel()
                live_errors.append(
                    max(
                        float(
                            (
                                graph_layer.keys[row, :, :valid]
                                - reference_layer.keys[row, :, :valid]
                            ).abs().max().item()
                        ),
                        float(
                            (
                                graph_layer.values[row, :, :valid]
                                - reference_layer.values[row, :, :valid]
                            ).abs().max().item()
                        ),
                    )
                )
        return {
            "max_logit_error": float((graph_live - reference).abs().max().item()),
            "max_source_kv_error": max(source_errors),
            "max_live_kv_error": max(live_errors),
            "graph_tokens": graph_predictions,
            "reference_tokens": reference_predictions,
            "tokens_match": graph_predictions == reference_predictions,
        }

    suffixes_first = [
        torch.randint(
            0,
            model.config.vocab_size,
            (length,),
            generator=generator,
            device=device,
        )
        for length in (17, 18)
    ]
    suffixes_second = [
        torch.randint(
            0,
            model.config.vocab_size,
            (length,),
            generator=generator,
            device=device,
        )
        for length in (22, 13)
    ]
    prefix_first = run_prefix_replay_case(suffixes_first)
    prefix_second = run_prefix_replay_case(suffixes_second)
    print(
        {
            "case": "prefix_copy_dynamic_replay",
            "rank": rank,
            "first": prefix_first,
            "second": prefix_second,
        },
        flush=True,
    )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
