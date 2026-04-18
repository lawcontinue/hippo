#!/bin/bash
# Hippo Cluster — 双机部署脚本
# 用法: 在第二台 Mac Mini 上运行此脚本
# 前置: 两台 Mac Mini 在同一局域网

set -e

echo "🦛 Hippo Cluster — 双机部署"
echo "============================"
echo ""

# 1. 检查 Python
echo "📋 Step 1/5: 检查环境..."
if ! command -v python3 &>/dev/null; then
    echo "❌ Python3 未安装，请先安装: brew install python3"
    exit 1
fi
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "   ✅ Python: $PYTHON_VERSION"

# 2. 克隆 Hippo（或从第一台机器 scp）
HIPPO_DIR="$HOME/hippo"
if [ -d "$HIPPO_DIR" ]; then
    echo "   ✅ Hippo 目录已存在: $HIPPO_DIR"
    cd "$HIPPO_DIR"
    git pull 2>/dev/null || echo "   ⚠️ git pull 失败（可能不是 git 仓库）"
else
    echo ""
    echo "📋 Step 2/5: 获取 Hippo 代码..."
    echo "   选择方式:"
    echo "   1) git clone（需要 GitHub 访问）"
    echo "   2) 从第一台 Mac Mini scp（推荐，最快）"
    echo ""
    read -p "   选择 [1/2]: " METHOD

    if [ "$METHOD" = "2" ]; then
        read -p "   第一台 Mac Mini IP: " HOST_IP
        read -p "   第一台 Mac Mini 用户名 [$(whoami)]: " HOST_USER
        HOST_USER=${HOST_USER:-$(whoami)}
        echo "   📥 从 $HOST_USER@$HOST_IP 同步..."
        scp -r "$HOST_USER@$HOST_IP:$HOME/hippo" "$HIPPO_DIR"
    else
        echo "   📥 git clone..."
        git clone https://github.com/deepsearch/hippo.git "$HIPPO_DIR" 2>/dev/null || {
            echo "   ⚠️ git clone 失败，请手动 clone 或使用 scp 方式"
            exit 1
        }
    fi
    cd "$HIPPO_DIR"
fi

# 3. 创建虚拟环境
echo ""
echo "📋 Step 3/5: 安装依赖..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -e ".[dev]" 2>/dev/null || pip install -e . 2>/dev/null || {
    echo "   安装核心依赖..."
    pip install fastapi uvicorn typer pydantic pyyaml llama-cpp-python aiohttp zeroconf requests
}
echo "   ✅ 依赖已安装"

# 4. 创建模型目录
mkdir -p ~/.hippo/models

# 5. 获取本机 IP
LOCAL_IP=$(python3 -c "
import socket
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.connect(('8.8.8.8', 80))
    print(s.getsockname()[0])
" 2>/dev/null || echo "未知")

echo ""
echo "📋 Step 4/5: 系统检测..."
echo "   本机 IP: $LOCAL_IP"
MEM_GB=$(python3 -c "
import subprocess
r = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
print(int(r.stdout.strip()) / (1024**3))
" 2>/dev/null || echo "16.0")
echo "   内存: ${MEM_GB} GB"

# 6. 验证 cluster 模块
echo ""
echo "📋 Step 5/5: 验证 cluster 模块..."
python3 -c "
from hippo.cluster.discovery import DiscoveryService
from hippo.cluster.worker import WorkerService, WorkerConfig
from hippo.cluster.gateway import GatewayService
from hippo.cluster.scheduler import Scheduler
from hippo.cluster.transport import Transport
print('   ✅ 所有 cluster 模块导入成功')
"

echo ""
echo "============================"
echo "✅ 部署完成！"
echo ""
echo "🚀 启动方式:"
echo ""
echo "  方案 A: 第一台 Mac Mini 启动 Gateway:"
echo "    cd $HIPPO_DIR && source .venv/bin/activate"
echo "    python -m hippo.cli gateway --host 0.0.0.0 --port 11434"
echo ""
echo "  方案 B: 第二台 Mac Mini 启动 Worker:"
echo "    cd $HIPPO_DIR && source .venv/bin/activate"
echo "    python -m hippo.cli worker --port 11435"
echo ""
echo "  或者手动指定 Gateway IP:"
echo "    python -m hippo.cli worker --gateway <GATEWAY_IP> --port 11435"
echo ""
echo "  查看集群状态:"
echo "    curl http://$LOCAL_IP:11434/cluster/status"
