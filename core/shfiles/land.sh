UAV_NAME=${UAV_NAME:-uav}
rostopic pub -1 /${UAV_NAME}/px4ctrl/takeoff_land quadrotor_msgs/TakeoffLand "takeoff_land_cmd: 2"
