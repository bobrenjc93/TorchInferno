# inference-bench Integration

TorchInferno exposes an OpenAI-compatible server so `inference-bench` can run it
beside `vllm` and `sglang`.

Copy `torchinferno.py` into the benchmark checkout:

```bash
cp integrations/inference_bench/torchinferno.py \
  /path/to/inference-bench/inference_bench/providers/torchinferno.py
```

Then update `inference_bench/providers/__init__.py` in that checkout so the lazy
provider import includes `torchinferno` next to `vllm` and `sglang`, and add the
provider to `config.yaml`:

```yaml
providers:
  - vllm
  - sglang
  - torchinferno
```

For `tensor_parallel_size > 1`, the provider starts TorchInferno with
`python -m torch.distributed.run` so inference-bench exercises the tensor
parallel server path instead of the pipeline fallback:

```bash
python -m torch.distributed.run --standalone --nproc-per-node 8 \
  -m torchinferno.openai_server \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --tensor-parallel-size 8 \
  --llama-parallelism tensor \
  --port 8000 \
  --trust-remote-code
```

The HTTP surface implements
`GET /v1/models` and streaming `POST /v1/chat/completions`, which are the
endpoints used by `inference-bench`.
