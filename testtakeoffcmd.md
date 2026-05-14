# UAV1 Takeoff Test Commands

本流程用于测试当前版本的自动起飞逻辑：

- `AUTO_TAKEOFF` 阶段使用 PX4 local position 作为起飞 setpoint 坐标系。
- 起飞目标为触发起飞瞬间 PX4 当前点正上方 `takeoff_height`。
- 起飞完成后切回原来的 VINS/px4ctrl 控制链路。

## 0. 所有终端通用环境

每个新终端先执行：

```bash
cd /home/uav/Desktop/uav_project/muav
source /opt/ros/noetic/setup.bash
source core/devel/setup.bash
export UAV_NAME=uav1
```

## 1. 编译确认

```bash
cd /home/uav/Desktop/uav_project/muav/core
catkin_make --pkg px4ctrl
```

## 2. 启动主流程

终端 1：

```bash
cd /home/uav/Desktop/uav_project/muav
UAV_NAME=uav1 ./core/shfiles/basectrl.sh
```

等待 MAVROS、VINS、px4ctrl 都启动完成。

## 3. 监控 PX4 与 VINS 位姿

终端 2：

```bash
cd /home/uav/Desktop/uav_project/muav
./core/shfiles/watch_px4_vins_pose.py --uav uav1 --rate 5
```

重点看同一张表里的：

```text
PX4      x y z yaw age
VINS     x y z yaw age
PX4-VINS dx dy dz dyaw dxy status
```

## 4. 监控 PX4 拒绝原因

终端 3：

```bash
rostopic echo /uav1/mavros/statustext/recv
```

如果出现 `ARM rejected` 或 `OFFBOARD rejected`，优先看这里。

另一个可选检查：

```bash
rostopic echo /uav1/mavros/state
```

确认：

```text
connected: True
armed: false/true
mode: OFFBOARD
```

## 5. 检查 PX4 local position

终端 4，可选：

```bash
cd /home/uav/Desktop/uav_project/muav
./core/shfiles/watch_px4_odom.sh --zero
```

或者只看频率：

```bash
rostopic hz /uav1/mavros/local_position/pose
```

## 6. 本机记录 rosbag

终端 5：

```bash
mkdir -p ~/uav_takeoff_logs

rosbag record -O ~/uav_takeoff_logs/uav1_takeoff_$(date +%Y%m%d_%H%M%S).bag --tcpnodelay \
/rosout_agg \
/uav1/mavros/state \
/uav1/mavros/extended_state \
/uav1/mavros/statustext/recv \
/uav1/mavros/local_position/pose \
/uav1/mavros/setpoint_position/local \
/uav1/mavros/setpoint_raw/attitude \
/uav1/mavros/imu/data \
/uav1/mavros/battery \
/uav1/vins_fusion/imu_propagate \
/uav1/vins_fusion/odometry \
/uav1/px4ctrl/takeoff_land
```

## 7. 触发起飞

确认以下条件后再起飞：

- 飞机落地且桨周围安全。
- `/uav1/mavros/local_position/pose` 有持续数据。
- `/uav1/vins_fusion/imu_propagate` 有持续数据。
- 遥控器开关处于 px4ctrl 要求的 hover/command 状态，摇杆居中。
- 终端 2 的 `age` 正常，通常应小于 `0.2s`。

终端 6：

```bash
cd /home/uav/Desktop/uav_project/muav
UAV_NAME=uav1 ./core/shfiles/takeoff.sh
```

等价直接命令：

```bash
rostopic pub -1 /uav1/px4ctrl/takeoff_land quadrotor_msgs/TakeoffLand "takeoff_land_cmd: 1"
```

## 8. 紧急降落

```bash
cd /home/uav/Desktop/uav_project/muav
UAV_NAME=uav1 ./core/shfiles/land.sh
```

等价直接命令：

```bash
rostopic pub -1 /uav1/px4ctrl/takeoff_land quadrotor_msgs/TakeoffLand "takeoff_land_cmd: 2"
```

## 9. 测试后提取关键日志

```bash
grep -RInE "px4ctrl|AUTO_TAKEOFF|AUTO_HOVER|ARM|Reject|rejected|OFFBOARD|statustext|VINS|PX4 local|WARN|ERROR" ~/.ros/log/latest | tail -n 300
```
rosrun mavros mavcmd -n /uav1/mavros long 246 1 0 0 0 0 0 0
## 10. 预期现象

起飞阶段：

- `/uav1/mavros/setpoint_position/local` 的 `x/y/yaw` 应接近触发起飞瞬间的 PX4 local pose。
- `z` 应从 PX4 起始高度逐渐爬升到 `px4_start_z + takeoff_height`。
- 当前配置默认 `takeoff_height: 0.5`，`takeoff_land_speed: 0.2`。

起飞完成后：

- px4ctrl 从 `AUTO_TAKEOFF` 切到 `AUTO_HOVER`。
- 后续悬停、规划和控制恢复使用原来的 VINS/px4ctrl 逻辑。

如果再次出现快速 yaw 旋转、明显水平漂移或 `ARM rejected`，立即降落并保留 rosbag 与日志。
