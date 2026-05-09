cmake_minimum_required(VERSION 3.0.2)
project(adv)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  quadrotor_msgs
  rospy
  std_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs quadrotor_msgs rospy std_msgs
)

catkin_install_python(PROGRAMS
  scripts/adv_ros_adapter
  scripts/adv_ros_e2e_smoke_test
  scripts/adv_ros_e2e_speed_test
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(FILES
  adv_module/inference_config.yaml
  adv_module/model_config.yaml
  adv_module/policy_checkpoint.pt
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
