#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${AGENTSERV_PORT:-4002}"
HOST="${AGENTSERV_HOST:-127.0.0.1}"
CONFIG="${AGENTSERV_CONFIG:-$SCRIPT_DIR/configs/tool_calling_agent.py}"
VENV_DIR="${AGENTSERV_VENV_DIR:-$SCRIPT_DIR/.venv}"
PYTHON_VERSION="${AGENTSERV_PYTHON_VERSION:-3.12}"
PYTHON_INTERPRETER="${AGENTSERV_PYTHON_INTERPRETER:-/opt/homebrew/opt/python@3.12/libexec/bin/python3}"
PYTHON_BIN="${AGENTSERV_PYTHON_BIN:-$VENV_DIR/bin/python}"
PID_FILE="$SCRIPT_DIR/.agentserv.pid"
SERVICE_LABEL="com.agentserv.gateway"

print_banner() {
    echo "════════════════════════════════════════"
    echo "  AgentServ — Agent Gateway"
    echo "════════════════════════════════════════"
}

check_deps() {
    if ! command -v uv &>/dev/null; then
        echo "uv required (brew install uv)"; exit 1
    fi
}

install_deps() {
    local requirements_file
    echo "Installing dependencies..."
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        uv venv --python "$PYTHON_INTERPRETER" "$VENV_DIR"
    elif ! "$VENV_DIR/bin/python" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"; then
        rm -rf "$VENV_DIR"
        uv venv --python "$PYTHON_INTERPRETER" "$VENV_DIR"
    fi
    requirements_file="$(mktemp)"
    grep -v '^graspologic' "$SCRIPT_DIR/requirements.txt" > "$requirements_file"
    UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv pip install --prerelease=allow -r "$requirements_file"
    rm -f "$requirements_file"
}

run_agentserv() {
    mkdir -p "$SCRIPT_DIR/workdir"
    env \
        OPENROUTER_API_BASE="${OPENROUTER_API_BASE:-http://127.0.0.1:4000/v1}" \
        OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-sk-aiserv-local-dev}" \
        ANTHROPIC_API_BASE="${ANTHROPIC_API_BASE:-http://127.0.0.1:4000}" \
        ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-sk-aiserv-local-dev}" \
        ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-sk-aiserv-local-dev}" \
        AGENTSERV_MCP_URL="${AGENTSERV_MCP_URL:-http://127.0.0.1:4001/hub/mcp}" \
        AGENTSERV_MEMORY_URL="${AGENTSERV_MEMORY_URL:-http://127.0.0.1:18800/stats}" \
        CRAWL4_AI_BASE_DIRECTORY="${CRAWL4_AI_BASE_DIRECTORY:-$SCRIPT_DIR/workdir}" \
        DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:-/opt/homebrew/opt/cairo/lib}" \
        "$PYTHON_BIN" agentserv.py --config "$CONFIG" --host "$HOST" --port "$PORT"
}

http_json() {
    local path="$1"
    "$PYTHON_BIN" - "$HOST" "$PORT" "$path" <<'PY'
import json
import sys
import urllib.request

host, port, path = sys.argv[1:4]
url = f"http://{host}:{port}{path}"
req = urllib.request.Request(url, headers={"Accept": "application/json"})
with urllib.request.urlopen(req, timeout=5) as response:
    payload = json.loads(response.read().decode("utf-8"))
print(json.dumps(payload, indent=2))
PY
}

is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        rm -f "$PID_FILE"
    fi
    if lsof -ti ":$PORT" >/dev/null 2>&1; then
        return 0
    fi
    if pgrep -f "agentserv\.py.*--port $PORT" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

find_agentserv_pids() {
    pgrep -f "agentserv\.py" 2>/dev/null || true
}

stop_all_processes() {
    local stopped=0
    local pid=""
    local uid=""
    local port_pids=""
    local stale=""

    uid="$(id -u)"
    if launchctl print "gui/$uid/$SERVICE_LABEL" &>/dev/null 2>&1; then
        launchctl bootout "gui/$uid/$SERVICE_LABEL" 2>/dev/null || true
        stopped=1
    fi

    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            stopped=1
        fi
        rm -f "$PID_FILE"
    fi

    stale="$(find_agentserv_pids)"
    if [ -n "$stale" ]; then
        echo "$stale" | xargs kill 2>/dev/null || true
        sleep 1
        stale="$(find_agentserv_pids)"
        if [ -n "$stale" ]; then
            echo "$stale" | xargs kill -9 2>/dev/null || true
        fi
        stopped=1
    fi

    port_pids="$(lsof -ti ":$PORT" 2>/dev/null || true)"
    if [ -n "$port_pids" ]; then
        echo "$port_pids" | xargs kill 2>/dev/null || true
        sleep 1
        port_pids="$(lsof -ti ":$PORT" 2>/dev/null || true)"
        if [ -n "$port_pids" ]; then
            echo "$port_pids" | xargs kill -9 2>/dev/null || true
        fi
        stopped=1
    fi

    if [ "$stopped" -eq 1 ]; then
        return 0
    fi
    return 1
}

start_service() {
    local mode="${1:-background}"

    print_banner
    check_deps
    install_deps
    mkdir -p "$SCRIPT_DIR/logs"

    if [ "$mode" = "foreground" ]; then
        echo "Starting in foreground on $HOST:$PORT"
        echo "  Dashboard: http://$HOST:$PORT/"
        echo "  Health:    ./agentserv.sh health"
        echo "  Status:    ./agentserv.sh status"
        echo ""
        stop_all_processes >/dev/null 2>&1 || true
        run_agentserv
        return
    fi

    if is_running; then
        echo "Gateway already running on $HOST:$PORT"
        return
    fi

    echo ""
    echo "Starting AgentServ on $HOST:$PORT"
    echo "  Dashboard: http://$HOST:$PORT/"
    echo "  Health:    ./agentserv.sh health"
    echo "  Status:    ./agentserv.sh status"
    echo ""

    stop_all_processes >/dev/null 2>&1 || true

    run_agentserv \
        >> "$SCRIPT_DIR/logs/agentserv.log" \
        2>> "$SCRIPT_DIR/logs/agentserv.error.log" &

    local local_pid
    local_pid=$!
    echo "$local_pid" > "$PID_FILE"
    echo "Started (PID: $local_pid)"
}

case "${1:-start}" in
    start)
        if [ "${2:-}" = "fg" ] || [ "${2:-}" = "--fg" ] || [ "${2:-}" = "--foreground" ]; then
            start_service foreground
        else
            start_service background
        fi
        ;;

    start-fg)
        start_service foreground
        ;;

    stop|stop-all)
        echo "Stopping AgentServ..."
        if stop_all_processes; then
            echo "Stopped"
        else
            echo "Not running"
        fi
        ;;

    restart)
        if [ "${2:-}" = "fg" ] || [ "${2:-}" = "--fg" ] || [ "${2:-}" = "--foreground" ]; then
            "$0" stop
            sleep 1
            "$0" start --foreground
        else
            $0 stop
            sleep 1
            $0 start
        fi
        ;;

    status)
        if is_running; then
            echo "AgentServ is running on $HOST:$PORT"
            find_agentserv_pids | while read -r p; do
                ps -o pid=,ppid=,lstart= -p "$p" 2>/dev/null | awk '{printf "  PID %-6s parent=%-6s started=%s %s %s %s %s\n", $1, $2, $3, $4, $5, $6, $7}'
            done
            echo ""
            http_json "/api/status" 2>/dev/null \
                || echo "  (API not responding yet)"
        else
            echo "AgentServ is not running"
        fi
        ;;

    health)
        echo "Health:"
        http_json "/health" 2>/dev/null \
            || echo "  (not reachable)"
        ;;

    install)
        check_deps
        install_deps
        echo "Dependencies installed in $VENV_DIR"
        ;;

    test)
        check_deps
        install_deps
        bash -n "$SCRIPT_DIR/agentserv.sh"
        plutil -lint "$SCRIPT_DIR/com.agentserv.gateway.plist"
        "$PYTHON_BIN" -m py_compile "$SCRIPT_DIR/agentserv.py"
        "$PYTHON_BIN" -m flake8 "$SCRIPT_DIR/agentserv.py" --max-line-length=120
        ;;

    enable)
        echo "Installing launchd service..."
        PLIST_PATH="$SCRIPT_DIR/com.agentserv.gateway.plist"
        mkdir -p "$SCRIPT_DIR/logs"

        launchctl bootout "gui/$(id -u)/$SERVICE_LABEL" 2>/dev/null || true
        stop_all_processes >/dev/null 2>&1 || true

        launchctl bootstrap "gui/$(id -u)" "$PLIST_PATH"

        echo "Service installed and started"
        echo "  Plist: $PLIST_PATH"
        echo "  Logs:  $SCRIPT_DIR/logs/"
        echo ""
        sleep 5
        if http_json "/health" >/dev/null 2>&1; then
            echo "Service is running: http://$HOST:$PORT"
        else
            echo "Service starting... check: launchctl print gui/$(id -u)/$SERVICE_LABEL"
        fi
        ;;

    disable)
        echo "Removing launchd service..."
        launchctl bootout "gui/$(id -u)/$SERVICE_LABEL" 2>/dev/null || true
        stop_all_processes >/dev/null 2>&1 || true
        echo "Service removed."
        ;;

    logs)
        echo "Recent logs:"
        echo ""
        echo "-- stdout --"
        tail -30 "$SCRIPT_DIR/logs/agentserv.log" 2>/dev/null || echo "  (no stdout log)"
        echo ""
        echo "-- stderr --"
        tail -30 "$SCRIPT_DIR/logs/agentserv.error.log" 2>/dev/null || echo "  (no stderr log)"
        ;;

    *)
        echo "Usage: $0 {start [--foreground]|start-fg|stop|stop-all|restart [--foreground]|status|health|install|test|enable|disable|logs}"
        exit 1
        ;;
esac
