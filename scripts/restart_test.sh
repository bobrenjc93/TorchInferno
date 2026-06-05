#!/bin/bash
# Kill existing servers, restart, wait for ready, then run test
set -e
cd /data/users/bobren/d/TorchInferno

echo "Killing existing servers..."
pkill -9 -f torchinferno 2>/dev/null || true
pkill -9 -f "torch.distributed" 2>/dev/null || true
sleep 3

echo "Starting server..."
rm -f /tmp/torchinferno_server.log
PYTHONPATH=src nohup python -m torchinferno.openai_server \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --tensor-parallel-size 8 \
    --port 8321 \
    --trust-remote-code \
    > /tmp/torchinferno_server.log 2>&1 &

echo "Waiting for server to be ready..."
for i in $(seq 1 120); do
    if grep -q "Listening" /tmp/torchinferno_server.log 2>/dev/null; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 5
done

if ! grep -q "Listening" /tmp/torchinferno_server.log 2>/dev/null; then
    echo "Server failed to start!"
    tail -30 /tmp/torchinferno_server.log
    exit 1
fi

# Show warmup logs
echo "=== Warmup logs ==="
grep -E "WARMUP|FI_PREFILL" /tmp/torchinferno_server.log | head -20

echo ""
echo "=== Sending test requests ==="
pip install aiohttp 2>/dev/null | tail -1
PYTHONPATH=src python scripts/test_ttft.py 16 2>&1

echo ""
echo "=== Server FI_PREFILL logs ==="
grep -E "FI_PREFILL" /tmp/torchinferno_server.log | tail -20
