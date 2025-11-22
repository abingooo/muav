sudo chmod 777 /dev/ttyACM0

sh shfiles/rspx4.sh

#imu 提高回传频率
rosrun mavros mavcmd long 511 105 4550 0 0 0 0 0 &sleep 1
rosrun mavros mavcmd long 511 31  4550 0 0 0 0 0 &sleep 1


rostopic hz mavros/imu/data_raw
 

#vins 外参校准
rostopic echo vins_fusion/imu_propagate

# 建图测试

sh shfiles/rspx4.sh

roslaunch ego_planner single_run_in_exp.launch

roslaunch ego_planner rviz.launch

# 起飞悬停
roslaunch px4ctrl run_ctrl.launch

sh shfiles/takeoff.sh

#egoplanner
sh shfiles/rspx4.sh
rostopic echo /vins_fusion/imu_propagate
roslaunch px4ctrl run_ctrl.launch
roslaunch ego_planner single_run_in_exp.launch
roslaunch ego_planner rviz.launch
sh shfiles/takeoff.sh



catkin_make
source  ./devel/setup.bash
sh shfiles/rspx4.sh
rosrun langguide langguide_node.py
rostopic echo /langguide/status

rostopic pub /lang_cmd std_msgs/String "Fly to the front of the tree"