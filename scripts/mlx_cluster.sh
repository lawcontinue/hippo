#!/bin/bash
# Hippo MLX 分布式推理启动脚本
# 
# 用法:
#   Mac Mini #1 (主节点): bash scripts/mlx_cluster.sh main
#   Mac Mini #2 (工作节点): bash scripts/mlx_cluster.sh worker 192.168.1.10

set -e

MODE="${1:-main}"
MAIN_IP="${2:-127.0.0.1}"
WORKER_IP="${3:-$(python3 -c "
import socket
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.connect(('8.8.8.8', 80))
    print(s.getsockname()[0])
")}"

MAIN_PORT=29500

echo "🦛 Hippo MLX Cluster"
echo "   Mode: $MODE"
echo "   Main IP: $MAIN_IP"
echo "   Worker IP: $WORKER_IP"
echo "   Port: $MAIN_PORT"
echo ""

if [ "$MODE" = "main" ]; then
    # 主节点 (rank=0)
    export MLX_RANK=0
    echo "Starting as MAIN (rank=0)"
    echo "Waiting for worker to connect..."
    
    # 主节点需要 hostfile
    cat > /tmp/hippo_hostfile.txt << EOF
$MAIN_IP
EOF
    
    export MLX_HOSTFILE=/tmp/hippo_hostfile.txt
    
elif [ "$MODE" = "worker" ]; then
    # 工作节点 (rank=1)
    export MLX_RANK=1
    echo "Starting as WORKER (rank=1)"
    echo "Connecting to main at $MAIN_IP..."
    
    # Worker 也需要 hostfile
    cat > /tmp/hippo_hostfile.txt << EOF
$MAIN_IP
EOF
    
    export MLX_HOSTFILE=/tmp/hippo_hostfile.txt
else
    echo "Usage: $0 main|worker [main_ip] [worker_ip]"
    exit 1
fi

echo "MLX_RANK=$MLX_RANK"
echo "MLX_HOSTFILE=$MLX_HOSTFILE"
echo ""
echo "✅ Environment configured"
echo ""
echo "Now run your MLX script with these env vars."
echo "Example:"
echo "  python3 your_inference_script.py"
