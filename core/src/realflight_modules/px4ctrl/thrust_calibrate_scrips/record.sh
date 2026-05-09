UAV_NAME=${UAV_NAME:-uav}
rosbag record --tcpnodelay /${UAV_NAME}/mavros/battery /${UAV_NAME}/mavros/setpoint_raw/attitude /${UAV_NAME}/traj_start_trigger
