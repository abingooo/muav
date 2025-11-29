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


cd lab/muav
catkin_make
source  ./devel/setup.bash
sh shfiles/rspx4.sh
rosrun langguide langguide_node.py
rosrun langguide lcmdpublisher.py
rostopic echo /langguide/status

rostopic pub /lang_cmd std_msgs/String "Fly to the front of the tree"



运行流程：先 catkin_make、source devel/setup.bash、启动 roscore，再分别 rosrun planner_plan_manage test_plan_point_listener 与 rosrun langguide test_plan_point_publisher.py，即可看到 20 次的闭环交互。


# 总流程
ssh uav0
cd lab/muav
catkin_make
source  ./devel/setup.bash

# 启动脚本
sh shfiles/rspx4.sh
# 提高IMU频率
rosrun mavros mavcmd long 511 105 4550 0 0 0 0 0 &sleep 1
rosrun mavros mavcmd long 511 31  4550 0 0 0 0 0 &sleep 1
# 查看IMU频率
rostopic hz mavros/imu/data_raw
# 查看VINS定位数据
rostopic echo vins_fusion/imu_propagate
# px4ctrl 
roslaunch px4ctrl run_ctrl.launch
# egoplanner
roslaunch ego_planner single_run_in_exp.launch
# rviz
roslaunch ego_planner rviz.launch
# 语言指导点
rosrun langguide langguide_node.py
# 控制指令发布
rosrun langguide lcmdpublisher.py
# 起飞节点
sh shfiles/takeoff.sh