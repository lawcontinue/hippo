#!/usr/bin/env bash
# start.sh — One-command Hippo Pipeline launcher
# Usage:
#   ./start.sh r0                    # Start R0 (pipeline mode)
#   ./start.sh r1                    # Start R1 (pipeline mode)
#   ./start.sh dflash                # Start DFlash single-machine
#   ./start.sh both                  # Start both R0+R1 via SSH
#   ./start.sh status                # Check running processes
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONUNBUFFERED=1

# Configurable via env vars
RANK0_HOST="${RANK0_HOST:-192.168.1.10}"
RANK1_HOST="${RANK1_HOST:-192.168.1.11}"
RANK1_USER="${RANK1_USER:-finance}"
PORT="${PORT:-9998}"
MODEL="${MODEL:-gemma-3-12b}"

usage() {
    echo "Usage: $0 {r0|r1|dflash|both|status} [--model MODEL] [--prompt PROMPT]"
    echo ""
    echo "Commands:"
    echo "  r0       Start Rank 0 (pipeline, local)"
    echo "  r1       Start Rank 1 (pipeline, local)"
    echo "  dflash   Start DFlash single-machine"
    echo "  both     Start R0 locally + R1 via SSH"
    echo "  status   Check running pipeline processes"
    echo ""
    echo "Environment:"
    echo "  RANK0_HOST  R0 IP (default: 192.168.1.10)"
    echo "  RANK1_HOST  R1 IP (default: 192.168.1.11)"
    echo "  RANK1_USER  R1 SSH user (default: finance)"
    echo "  PORT        Port (default: 9998)"
    echo "  MODEL       Model (default: gemma-3-12b)"
}

start_r0() {
    local prompt="${1:-Hello}"
    echo "🚀 Starting R0 — model=$MODEL, port=$PORT"
    cd "$DIR"
    python3 sharded_inference.py --rank 0 --host "$RANK0_HOST" --port "$PORT" --rank0-host "$RANK0_HOST" --prompt "$prompt"
}

start_r1() {
    echo "🚀 Starting R1 — connecting to R0 at $RANK0_HOST:$PORT"
    cd "$DIR"
    python3 sharded_inference.py --rank 1 --host 0.0.0.0 --port "$PORT" --rank0-host "$RANK0_HOST"
}

start_r1_remote() {
    echo "🚀 Starting R1 via SSH — $RANK1_USER@$RANK1_HOST"
    ssh "$RANK1_USER@$RANK1_HOST" \
        "cd ~/hippo/pipeline && PYTHONUNBUFFERED=1 python3 sharded_inference.py --rank 1 --host 0.0.0.0 --port $PORT --rank0-host $RANK0_HOST" \
        &>/tmp/r1_serve.log &
    local pid=$!
    echo "   R1 PID: $pid, log: /tmp/r1_serve.log"
    echo "$pid" > /tmp/r1_serve.pid
    sleep 2
    if kill -0 "$pid" 2>/dev/null; then
        echo "✅ R1 started successfully"
    else
        echo "❌ R1 failed to start. Check /tmp/r1_serve.log"
        exit 1
    fi
}

start_dflash() {
    echo "🚀 Starting DFlash — model=$MODEL"
    cd "$DIR"
    python3 rank0_dflash.py
}

start_both() {
    local prompt="${1:-Hello}"
    echo "🚀 Starting dual-machine pipeline"
    echo ""

    # Kill any existing processes
    kill_pipeline 2>/dev/null || true

    # Start R1 remotely
    start_r1_remote
    echo ""

    # Start R0 locally
    start_r0 "$prompt"
}

kill_pipeline() {
    # Kill local R0
    pkill -f "sharded_inference.py.*--rank 0" 2>/dev/null || true
    # Kill remote R1
    local pid
    pid=$(cat /tmp/r1_serve.pid 2>/dev/null || echo "")
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
    fi
    # Kill any leftover port listeners
    lsof -ti :$((PORT)),:$((PORT+1)) 2>/dev/null | xargs kill 2>/dev/null || true
    echo "🛑 Pipeline stopped"
}

check_status() {
    echo "📊 Hippo Pipeline Status"
    echo ""

    # Check R0
    if pgrep -f "sharded_inference.py.*--rank 0" > /dev/null 2>&1; then
        echo "  R0: ✅ running"
    else
        echo "  R0: ⬚ not running"
    fi

    # Check R1 local
    if pgrep -f "sharded_inference.py.*--rank 1" > /dev/null 2>&1; then
        echo "  R1 (local): ✅ running"
    else
        echo "  R1 (local): ⬚ not running"
    fi

    # Check R1 remote
    local pid
    pid=$(cat /tmp/r1_serve.pid 2>/dev/null || echo "")
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "  R1 (remote): ✅ running (PID $pid)"
        echo "    Last log:"
        tail -3 /tmp/r1_serve.log 2>/dev/null | sed 's/^/      /'
    else
        echo "  R1 (remote): ⬚ not running"
    fi

    # Check DFlash
    if pgrep -f "rank0_dflash.py" > /dev/null 2>&1; then
        echo "  DFlash: ✅ running"
    else
        echo "  DFlash: ⬚ not running"
    fi

    # Check ports
    echo ""
    echo "  Ports:"
    for p in $PORT $((PORT+1)); do
        local listener
        listener=$(lsof -ti :$p 2>/dev/null || true)
        if [ -n "$listener" ]; then
            echo "    :$p — ✅ listening (PID: $(echo $listener | tr '\n' ' '))"
        else
            echo "    :$p — ⬚ free"
        fi
    done
}

# Parse args
CMD="${1:-}"
shift || true

case "$CMD" in
    r0)     start_r0 "${1:-Hello}" ;;
    r1)     start_r1 ;;
    dflash) start_dflash ;;
    both)   start_both "${1:-Hello}" ;;
    status) check_status ;;
    stop)   kill_pipeline ;;
    help|--help|-h) usage ;;
    *)      echo "Unknown command: $CMD"; usage; exit 1 ;;
esac
