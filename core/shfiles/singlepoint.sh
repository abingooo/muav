#!/bin/sh
set -eu

if [ $# -ne 1 ]; then
  echo "Usage: $0 \"x y z\" (or \"x,y,z\")" >&2
  exit 1
fi

input="$1"
old_ifs=$IFS
case "$input" in
  *,*)
    IFS=',' set -- $input
    ;;
  *)
    IFS=' ' set -- $input
    ;;
esac
IFS=$old_ifs

if [ $# -ne 3 ] || [ -z "${1:-}" ] || [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
  echo "Invalid position format. Provide coordinates as \"x y z\" or \"x,y,z\"." >&2
  exit 1
fi

x=$1
y=$2
z=$3
UAV_NAME=${UAV_NAME:-uav}

rostopic pub -1 /${UAV_NAME}/toplan/single_plan_point geometry_msgs/PoseStamped "header:
  stamp: now
  frame_id: 'world'
pose:
  position: {x: ${x}, y: ${y}, z: ${z}}
  orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}"
