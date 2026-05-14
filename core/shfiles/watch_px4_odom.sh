#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

source /opt/ros/noetic/setup.bash
source "${CORE_DIR}/devel/setup.bash"

UAV_NAME=${UAV_NAME:-uav1}

exec python3 "${SCRIPT_DIR}/watch_px4_odom.py" --uav "${UAV_NAME}" "$@"
