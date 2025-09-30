#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="cctv.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"

install_service() {
  sudo tee "$SERVICE_PATH" >/dev/null <<EOF
[Unit]
Description=Raspberry Pi CCTV (tmux)
After=multi-user.target graphical.target network-online.target
Wants=graphical.target network-online.target

[Service]
Type=simple
User=jecon
Group=jecon
Environment=PYTHONUNBUFFERED=1
Environment=DISPLAY=:0
Environment=XAUTHORITY=/home/jecon/.Xauthority
Environment=XDG_RUNTIME_DIR=/run/user/1000
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
WorkingDirectory=/home/jecon/yolo/DNN_RECOGNISE
ExecStart=/home/jecon/yolo/DNN_RECOGNISE/start_cctv.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

  sudo systemctl daemon-reload
  sudo systemctl enable "$SERVICE_NAME"
  echo "Service installed and enabled: $SERVICE_NAME"
}

start_service() { sudo systemctl start "$SERVICE_NAME"; }
stop_service() { sudo systemctl stop "$SERVICE_NAME"; }
restart_service() { sudo systemctl restart "$SERVICE_NAME"; }
status_service() { sudo systemctl status "$SERVICE_NAME" --no-pager || true; }
logs_service() { journalctl -u "$SERVICE_NAME" -e -n 200 --no-pager; }

case "${1:-}" in
  install) install_service ;;
  start) start_service ;;
  stop) stop_service ;;
  restart) restart_service ;;
  status) status_service ;;
  logs) logs_service ;;
  *)
    echo "Usage: $0 {install|start|stop|restart|status|logs}"
    exit 1
    ;;
esac


