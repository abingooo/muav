#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${CORE_DIR}/.." && pwd)"
CONFIG_FILE="${PROJECT_DIR}/core/src/groundctrl/config/groundctrl.yaml"

readarray -t GROUND_ENV < <(
  python3 - "${CONFIG_FILE}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

ground = config["ground"]
print(ground["ip"])
print(ground["ros_master_uri"])
PY
)

GROUND_IP="${GROUND_ENV[0]}"
GROUND_MASTER_URI="${GROUND_ENV[1]}"

kill_by_pattern() {
  local pattern="$1"
  local pid
  local pids=()
  while IFS= read -r pid; do
    if [[ -n "${pid}" && "${pid}" != "$$" && "${pid}" != "${BASHPID}" && "${pid}" != "${PPID}" ]]; then
      pids+=("${pid}")
    fi
  done < <(pgrep -f "${pattern}" 2>/dev/null || true)

  if [ "${#pids[@]}" -eq 0 ]; then
    return 0
  fi

  kill "${pids[@]}" 2>/dev/null || true
  sleep 1
  local pid
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
}

source /opt/ros/noetic/setup.bash
if [ -f "${CORE_DIR}/devel/setup.bash" ]; then
  source "${CORE_DIR}/devel/setup.bash"
fi

export ROS_MASTER_URI="${GROUND_MASTER_URI}"
export ROS_IP="${GROUND_IP}"
unset ROS_HOSTNAME

COMMAND="${1:-sync}"
if [[ "${COMMAND}" != "sync" && "${COMMAND}" != "start" && "${COMMAND}" != "rviz" && "${COMMAND}" != "start-rviz" && "${COMMAND}" != "env" && "${COMMAND}" != "status" && "${COMMAND}" != "stop" ]]; then
  COMMAND="sync"
elif [ "$#" -gt 0 ]; then
  shift
fi

case "${COMMAND}" in
  sync|start)
    exec roslaunch groundctrl ground_sync.launch \
      ground_ip:="${GROUND_IP}" \
      ground_master_uri:="${GROUND_MASTER_URI}" \
      "$@"
    ;;
  rviz|start-rviz)
    exec roslaunch groundctrl ground_rviz.launch \
      ground_ip:="${GROUND_IP}" \
      ground_master_uri:="${GROUND_MASTER_URI}" \
      "$@"
    ;;
  env)
    printf 'ROS_MASTER_URI=%s\n' "${ROS_MASTER_URI}"
    printf 'ROS_IP=%s\n' "${ROS_IP}"
    ;;
  status)
    rosnode list | grep -E '^/(ground_master_discovery|ground_master_sync|ground_state_monitor|ground_rviz)$' || true
    ;;
  stop)
    rosnode kill /ground_master_sync /ground_master_discovery /ground_state_monitor /ground_rviz 2>/dev/null || true
    kill_by_pattern 'roslaunch groundctrl ground_sync.launch'
    kill_by_pattern 'roslaunch groundctrl ground_rviz.launch'
    kill_by_pattern 'fkie_master_sync/master_sync.*__name:=ground_master_sync'
    kill_by_pattern 'fkie_master_discovery/master_discovery.*__name:=ground_master_discovery'
    kill_by_pattern 'ground_state_monitor.py.*__name:=ground_state_monitor'
    kill_by_pattern 'rviz.*__name:=ground_rviz'
    ;;
  *)
    printf 'Usage: %s [sync|rviz|env|status|stop] [roslaunch args...]\n' "$0" >&2
    exit 2
    ;;
esac
