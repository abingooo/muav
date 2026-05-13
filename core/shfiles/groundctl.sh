#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
GROUNDCTL_ARGS=("$@")

set --
source /opt/ros/noetic/setup.bash
if [ -f "${CORE_DIR}/devel/setup.bash" ]; then
  source "${CORE_DIR}/devel/setup.bash"
fi

exec rosrun groundctrl groundctl.py "${GROUNDCTL_ARGS[@]}"
