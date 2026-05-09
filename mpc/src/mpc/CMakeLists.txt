cmake_minimum_required(VERSION 3.0.2)
project(mpc)

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
  scripts/mpc_ros_adapter
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(FILES
  mpc_module/model_config.yaml
  mpc_module/mpc_config.yaml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
