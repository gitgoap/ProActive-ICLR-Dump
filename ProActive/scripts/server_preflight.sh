#!/usr/bin/env bash
# ProActive Server Preflight Script
# Run this on your remote server and paste the output back to the agent.

echo "========================================"
echo "ProActive Server Preflight Report"
echo "Date: $(date -u)"
echo "Hostname: $(hostname)"
echo "========================================"

echo ""
echo "--- System & CPU ---"
uname -a
echo "Total RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "Available RAM: $(free -h | awk '/^Mem:/{print $7}')"

echo ""
echo "--- GPU Overview (nvidia-smi) ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "ERROR: nvidia-smi not found. GPUs may not be accessible."
fi

echo ""
echo "--- GPU Topology ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi topo -m 2>/dev/null || echo "Topology command not supported."
fi

echo ""
echo "--- Storage Availability ---"
df -h . /tmp /var/tmp 

echo ""
echo "--- Python Environment ---"
python3 --version 2>&1 || echo "python3 not found"
echo "pip packages:"
pip list | grep -E "torch|transformers|xformers|flash-attn|accelerate|qwen|torchvision|timm" || echo "No key packages found."

echo ""
echo "--- CUDA Toolkit (nvcc) ---"
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "nvcc not found in PATH."
fi

echo ""
echo "========================================"
echo "Preflight Complete"
echo "Please copy this output and paste it to Antigravity."
echo "========================================"
