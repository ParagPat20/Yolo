#!/usr/bin/env bash

# Start CCTV only when a GUI display is available for preview
# Usage: run this at boot via systemd or rc.local

set -euo pipefail

# Configuration
# Force project directory explicitly as requested
PROJECT_DIR="/home/jecon/yolo/DNN_RECOGNISE"
cd "$PROJECT_DIR"
SCRIPT="$PROJECT_DIR/src/cctv_system.py"

# Prefer existing DISPLAY, else try :0 (common on Raspberry Pi desktop)
ACTIVE_DISPLAY="${DISPLAY:-:0}"

# Determine XAUTHORITY for the active desktop user (best-effort)
detect_xauthority() {
  # Try to detect the user owning the GUI session
  local gui_user
  gui_user=$(who | awk '/(:[0-9]|tty)/ {print $1; exit}') || true
  if [[ -n "${gui_user:-}" && -f "/home/$gui_user/.Xauthority" ]]; then
    echo "/home/$gui_user/.Xauthority"
    return 0
  fi
  # Fallback to current user's Xauthority if present
  if [[ -n "${HOME:-}" && -f "$HOME/.Xauthority" ]]; then
    echo "$HOME/.Xauthority"
    return 0
  fi
  # Otherwise leave empty (might still work for Wayland)
  echo ""
}

XAUTHORITY_PATH="$(detect_xauthority)"

# Check if GUI display is ready by probing xset
is_gui_ready() {
  if command -v xset >/dev/null 2>&1; then
    if [[ -n "$XAUTHORITY_PATH" ]]; then
      DISPLAY="$ACTIVE_DISPLAY" XAUTHORITY="$XAUTHORITY_PATH" xset -q >/dev/null 2>&1 && return 0
    else
      DISPLAY="$ACTIVE_DISPLAY" xset -q >/dev/null 2>&1 && return 0
    fi
  fi
  # If xset not available, try a generic X11 socket probe
  [[ -S "/tmp/.X11-unix/${ACTIVE_DISPLAY#:}" ]] && return 0
  return 1
}

# Wait up to 90s for GUI to be ready (exit early if not desired)
TIMEOUT_SEC=90
SLEEP_SEC=3
elapsed=0
while (( elapsed < TIMEOUT_SEC )); do
  if is_gui_ready; then
    echo "[start_cctv] GUI detected on DISPLAY=$ACTIVE_DISPLAY"
    break
  fi
  echo "[start_cctv] GUI not ready yet; waiting... ($elapsed/${TIMEOUT_SEC}s)"
  sleep "$SLEEP_SEC"
  elapsed=$((elapsed + SLEEP_SEC))
done

if ! is_gui_ready; then
  echo "[start_cctv] GUI not available; not starting CCTV preview. Exiting."
  exit 0
fi

export DISPLAY="$ACTIVE_DISPLAY"
[[ -n "$XAUTHORITY_PATH" ]] && export XAUTHORITY="$XAUTHORITY_PATH"

cd "$PROJECT_DIR"
echo "[start_cctv] Starting CCTV in tmux session 'cctv'..."
if command -v tmux >/dev/null 2>&1; then
  # Create or reuse session
  tmux has-session -t cctv >/dev/null 2>&1 || tmux new-session -d -s cctv
  tmux send-keys -t cctv "cd '$PROJECT_DIR' && DISPLAY='$ACTIVE_DISPLAY' ${XAUTHORITY:+XAUTHORITY='$XAUTHORITY_PATH'} python3 -u '$SCRIPT'" C-m
  echo "[start_cctv] Launched in tmux. Attach with: tmux attach -t cctv"
  # Keep this process alive while the tmux session exists so systemd doesn't
  # tear down the cgroup and kill tmux. When session ends, exit.
  while tmux has-session -t cctv >/dev/null 2>&1; do
    sleep 2
  done
  echo "[start_cctv] tmux session 'cctv' ended. Exiting."
  exit 0
else
  echo "[start_cctv] tmux not found; running in foreground."
  exec python3 -u "$SCRIPT"
fi


