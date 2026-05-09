UAV_NAME=${UAV_NAME:-uav}
rosbag record --tcpnodelay \
/${UAV_NAME}/vins_estimator/extrinsic \
/${UAV_NAME}/vins_estimator/keyframe_point \
/${UAV_NAME}/vins_estimator/keyframe_pose \
/${UAV_NAME}/vins_estimator/margin_cloud \
/${UAV_NAME}/vins_estimator/odometry \
/${UAV_NAME}/vins_fusion/camera_pose \
/${UAV_NAME}/vins_fusion/camera_pose_visual \
/${UAV_NAME}/vins_fusion/extrinsic \
/${UAV_NAME}/vins_fusion/image_track \
/${UAV_NAME}/vins_fusion/imu_propagate \
/${UAV_NAME}/vins_fusion/key_poses \
/${UAV_NAME}/vins_fusion/keyframe_point \
/${UAV_NAME}/vins_fusion/keyframe_pose \
/${UAV_NAME}/vins_fusion/margin_cloud \
/${UAV_NAME}/vins_fusion/odometry \
/${UAV_NAME}/vins_fusion/path \
/${UAV_NAME}/vins_fusion/point_cloud
