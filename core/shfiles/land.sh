#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

source /opt/ros/noetic/setup.bash
source "${CORE_DIR}/devel/setup.bash"

UAV_NAME=${UAV_NAME:-uav}
rostopic pub -1 /${UAV_NAME}/px4ctrl/takeoff_land quadrotor_msgs/TakeoffLand "takeoff_land_cmd: 2"
