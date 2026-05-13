cmake_minimum_required(VERSION 3.0.2)
project(groundctrl)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  quadrotor_msgs
  rospy
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs quadrotor_msgs rospy std_msgs
)

catkin_install_python(PROGRAMS
  scripts/ground_state_monitor.py
  scripts/ground_odom_watch.py
  scripts/ground_web.py
  scripts/groundctl.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch docs
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
