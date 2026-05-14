#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

source /opt/ros/noetic/setup.bash
source "${CORE_DIR}/devel/setup.bash"

UAV_NAME=${UAV_NAME:-uav1}
FCU_URL=${FCU_URL:-/dev/ttyACM0:57600}
SERIAL_DEV="${FCU_URL%%:*}"

export UAV_NAME
export ROS_NAMESPACE="/${UAV_NAME}"

cleanup() {
  if [[ -n "${MAVROS_PID:-}" ]]; then
    kill "${MAVROS_PID}" 2>/dev/null || true
  fi
  if [[ -n "${ROSCORE_PID:-}" ]]; then
    kill "${ROSCORE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if ! rosnode list >/dev/null 2>&1; then
  echo "[px4_odom_only] starting roscore..."
  roscore >/tmp/px4_odom_only_roscore.log 2>&1 &
  ROSCORE_PID=$!
  until rosnode list >/dev/null 2>&1; do
    sleep 0.2
  done
else
  echo "[px4_odom_only] using existing roscore."
fi

if [[ -e "${SERIAL_DEV}" ]]; then
  sudo chmod 666 "${SERIAL_DEV}" || true
fi

echo "[px4_odom_only] starting mavros for ${UAV_NAME}, fcu_url=${FCU_URL}"
roslaunch px4ctrl mavros_px4_namespaced.launch fcu_url:="${FCU_URL}" &
MAVROS_PID=$!

echo "[px4_odom_only] waiting for /${UAV_NAME}/mavros/local_position/pose ..."
until rostopic list 2>/dev/null | grep -qx "/${UAV_NAME}/mavros/local_position/pose"; do
  sleep 0.5
done

python3 "${SCRIPT_DIR}/watch_px4_odom.py" --uav "${UAV_NAME}" "$@"
