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
