#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${CORE_DIR}/.." && pwd)"
CONFIG_FILE="${PROJECT_DIR}/core/src/groundctrl/config/groundctrl.yaml"
WEB_ARGS=("$@")

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

set --
set +u
source /opt/ros/noetic/setup.bash
if [ -f "${CORE_DIR}/devel/setup.bash" ]; then
  source "${CORE_DIR}/devel/setup.bash"
fi
set -u

export ROS_MASTER_URI="${GROUND_MASTER_URI}"
export ROS_IP="${GROUND_IP}"
unset ROS_HOSTNAME

exec rosrun groundctrl ground_web.py "${WEB_ARGS[@]}"
