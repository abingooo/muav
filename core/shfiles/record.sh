UAV_NAME=${UAV_NAME:-uav}
rosbag record --tcpnodelay \
/${UAV_NAME}/drone_0_ego_planner_node/goal_point \
/${UAV_NAME}/ego_planner_node/global_list \
/${UAV_NAME}/drone_0_ego_planner_node/optimal_list \
/${UAV_NAME}/ego_planner_node/a_star_list \
/${UAV_NAME}/drone_0_ego_planner_node/init_list \
/${UAV_NAME}/drone_0_odom_visualization/path \
/${UAV_NAME}/drone_0_ego_planner_node/grid_map/occupancy_inflate \
/${UAV_NAME}/drone_0_odom_visualization/robot \
/${UAV_NAME}/vins_fusion/path \
/${UAV_NAME}/vins_fusion/odometry \
/${UAV_NAME}/camera/infra1/image_rect_raw \
/${UAV_NAME}/position_cmd \
