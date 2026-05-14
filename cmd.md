# core
cd /home/uav/Desktop/uav_project/muav
roscore

cd /home/uav/Desktop/uav_project/muav
./core/shfiles/basectrl.sh

cd /home/uav/Desktop/uav_project/muav
roslaunch swarm_position_bridge uav_vins_sync.launch

cd /home/uav/Desktop/uav_project/muav
./core/shfiles/takeoff.sh

# mpc: run on current UAV. UAV_NAME decides enemy.
cd /home/uav/Desktop/uav_project/muav/mpc
source devel/setup.bash
UAV_NAME=uav2 roslaunch mpc mpc.launch

# adv: run on current UAV. enemy_uav must be set.
cd /home/uav/Desktop/uav_project/muav/adv
source devel/setup.bash
UAV_NAME=uav0 roslaunch adv adv.launch enemy_uav:=uav2

# adv backup example: uav2 is enemy, so uav4 backs defender_2.
cd /home/uav/Desktop/uav_project/muav/adv
source devel/setup.bash
UAV_NAME=uav4 roslaunch adv adv.launch enemy_uav:=uav2



# test px4ctrl
• 可以按“三个终端”快速调。

  终端 1：看关键日志

  cd /home/uav/Desktop/uav_project/muav
  source /opt/ros/noetic/setup.bash
  source core/devel/setup.bash

  tail -f ~/.ros/log/latest/rosout.log | rg --line-buffered "px4ctrl|PX4 statustext|ARM rejected|Latest PX4 statustext|local pose|AUTO_TAKEOFF|OFFBOARD"

  终端 2：看实时 PX4 状态

  source /opt/ros/noetic/setup.bash
  source /home/uav/Desktop/uav_project/muav/core/devel/setup.bash

  rostopic echo /uav1/mavros/state

  重点看：

  connected: true
  mode: "OFFBOARD"
  armed: true/false

  另开一个也可以看 PX4 原始拒绝原因：

  rostopic echo /uav1/mavros/statustext/recv

  终端 3：发起飞

  确认安全后再执行：

  cd /home/uav/Desktop/uav_project/muav
  UAV_NAME=uav1 ./core/shfiles/takeoff.sh

  如果失败，现在日志里应该会直接出现类似：

  ARM rejected by PX4! mav_result=2(DENIED)
  Latest PX4 statustext ... text="Preflight Fail: ..."

  如果是坐标不一致，会出现：

  Reject AUTO_TAKEOFF. VINS odom and PX4 local position are not aligned.

  快速看 VINS 和 PX4 local pose 是否差太多：

  rostopic echo -n1 /uav1/vins_fusion/imu_propagate/pose/pose/position
  rostopic echo -n1 /uav1/mavros/local_position/pose/pose/position

  如果刚改完代码还没重启，先重启 px4ctrl/basectrl，否则跑的还是旧二进制。调试时优先看这条聚合命令：

  rg -n "PX4 statustext|ARM rejected|Latest PX4 statustext|local pose|Reject AUTO_TAKEOFF|AUTO_TAKEOFF|OFFBOARD" ~/.ros/log/latest/rosout.log