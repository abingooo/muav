#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

source /opt/ros/noetic/setup.bash
source "${CORE_DIR}/devel/setup.bash"

sudo chmod 777 /dev/ttyACM0 & sleep 2;
export UAV_NAME=${UAV_NAME:-uav}
export ROS_NAMESPACE=/${UAV_NAME}
export TF_REMAP_ARGS="/tf:=tf /tf_static:=tf_static"
export RS_TF_PREFIX=${RS_TF_PREFIX:-${UAV_NAME}_camera}
# roslaunch realsense2_camera rs_camera.launch & sleep 10;
roslaunch realsense2_camera rs_camera.launch align_depth:=true tf_prefix:=${RS_TF_PREFIX} ${TF_REMAP_ARGS} & sleep 10;
roslaunch px4ctrl mavros_px4_namespaced.launch & sleep 10;

rosrun mavros mavcmd -n /${UAV_NAME}/mavros long 511 105 4550 0 0 0 0 0 &sleep 1;
rosrun mavros mavcmd -n /${UAV_NAME}/mavros long 511 31  4550 0 0 0 0 0 &sleep 1;

roslaunch vins fast_drone_250.launch ${TF_REMAP_ARGS} & sleep 10;

roslaunch px4ctrl run_ctrl.launch ${TF_REMAP_ARGS} & sleep 1;

roslaunch ego_planner single_run_in_exp.launch ${TF_REMAP_ARGS} & sleep 2;

rosrun langguide langguide_node.py

wait;
