#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./core/shfiles/groundweb_service.sh install [--host 0.0.0.0] [--port 8080] [--enable] [--start]
  ./core/shfiles/groundweb_service.sh uninstall
  ./core/shfiles/groundweb_service.sh start|stop|restart|status|enable|disable
  ./core/shfiles/groundweb_service.sh logs [-f]
  ./core/shfiles/groundweb_service.sh unit

This installs a systemd user service:
  muav-groundweb.service

Notes:
  - Default host is 0.0.0.0 so the panel is reachable from LAN.
  - Use "enable" to start the service automatically after user login.
  - To start before login, run: sudo loginctl enable-linger $USER
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${CORE_DIR}/.." && pwd)"
GROUNDWEB_SH="${SCRIPT_DIR}/groundweb.sh"

SERVICE_NAME="muav-groundweb"
HOST="0.0.0.0"
PORT="8080"
DO_ENABLE=0
DO_START=0
FOLLOW_LOGS=0

CONFIG_HOME="${XDG_CONFIG_HOME:-${HOME}/.config}"
USER_SYSTEMD_DIR="${CONFIG_HOME}/systemd/user"
UNIT_FILE="${USER_SYSTEMD_DIR}/${SERVICE_NAME}.service"

COMMAND="${1:-}"
if [ "$#" -gt 0 ]; then
  shift
fi

while [ "$#" -gt 0 ]; do
  case "$1" in
    --host)
      HOST="${2:-}"
      shift 2
      ;;
    --host=*)
      HOST="${1#--host=}"
      shift
      ;;
    --port)
      PORT="${2:-}"
      shift 2
      ;;
    --port=*)
      PORT="${1#--port=}"
      shift
      ;;
    --enable)
      DO_ENABLE=1
      shift
      ;;
    --start)
      DO_START=1
      shift
      ;;
    -f|--follow)
      FOLLOW_LOGS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf '[groundweb_service] unknown argument: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

need_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf '[groundweb_service] required command not found: %s\n' "$1" >&2
    exit 1
  fi
}

systemctl_user() {
  SYSTEMD_PAGER=cat systemctl --user "$@"
}

write_unit() {
  mkdir -p "${USER_SYSTEMD_DIR}"
  cat > "${UNIT_FILE}" <<UNIT
[Unit]
Description=MUAV Ground Web Control Panel
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${PROJECT_DIR}
Environment=PYTHONUNBUFFERED=1
ExecStart=${GROUNDWEB_SH} --host ${HOST} --port ${PORT}
Restart=on-failure
RestartSec=2
KillMode=control-group

[Install]
WantedBy=default.target
UNIT
}

require_unit() {
  if [ ! -f "${UNIT_FILE}" ]; then
    printf '[groundweb_service] service is not installed: %s\n' "${UNIT_FILE}" >&2
    printf '[groundweb_service] run: %s install --enable --start\n' "$0" >&2
    exit 1
  fi
}

case "${COMMAND}" in
  install)
    need_command systemctl
    write_unit
    systemctl_user daemon-reload
    printf '[groundweb_service] installed %s\n' "${UNIT_FILE}"
    printf '[groundweb_service] ExecStart=%s --host %s --port %s\n' "${GROUNDWEB_SH}" "${HOST}" "${PORT}"
    if [ "${DO_ENABLE}" -eq 1 ]; then
      systemctl_user enable "${SERVICE_NAME}.service"
    fi
    if [ "${DO_START}" -eq 1 ]; then
      systemctl_user restart "${SERVICE_NAME}.service"
    fi
    ;;
  uninstall)
    need_command systemctl
    systemctl_user stop "${SERVICE_NAME}.service" >/dev/null 2>&1 || true
    systemctl_user disable "${SERVICE_NAME}.service" >/dev/null 2>&1 || true
    rm -f "${UNIT_FILE}"
    systemctl_user daemon-reload
    printf '[groundweb_service] uninstalled %s\n' "${UNIT_FILE}"
    ;;
  start|stop|restart|status|enable|disable)
    need_command systemctl
    require_unit
    systemctl_user "${COMMAND}" "${SERVICE_NAME}.service"
    ;;
  logs)
    require_unit
    if [ "${FOLLOW_LOGS}" -eq 1 ]; then
      journalctl --user -u "${SERVICE_NAME}.service" -n 80 -f
    else
      journalctl --user -u "${SERVICE_NAME}.service" -n 120 --no-pager
    fi
    ;;
  unit)
    require_unit
    cat "${UNIT_FILE}"
    ;;
  -h|--help|"")
    usage
    ;;
  *)
    printf '[groundweb_service] unknown command: %s\n\n' "${COMMAND}" >&2
    usage >&2
    exit 2
    ;;
esac
