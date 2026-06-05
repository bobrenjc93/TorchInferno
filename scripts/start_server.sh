#!/bin/bash
# Start the torchinferno OpenAI server on 8xH100
pkill -f "torchinferno.openai_server" 2>/dev/null
sleep 2

cd /data/users/bobren/d/TorchInferno
PYTHONPATH=src python -m torchinferno.openai_server \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --tensor-parallel-size 8 \
    --port 8321 \
    --trust-remote-code \
    2>&1 | tee /tmp/torchinferno_server.log
