# Groundctrl TODO

## 已完成的需求拆分

- [x] 在 ground 本机建立 `Desktop/uav_project/muav` 工程基线。
- [x] 确认 UAV SSH host：`uav0`、`uav1`、`uav2`。
- [x] 确认关键控制话题：
  - 起飞/降落：`/{uav}/px4ctrl/takeoff_land`，类型 `quadrotor_msgs/TakeoffLand`。
  - EGO 单点目标：`/{uav}/toplan/single_plan_point`，类型 `geometry_msgs/PoseStamped`。
  - EGO 航点完成：`/{uav}/toplan/waypoint_done`，类型 `std_msgs/String`。
  - PX4CTRL 位置/速度控制：`/{uav}/position_cmd`，类型 `quadrotor_msgs/PositionCommand`。
  - 里程计：`/{uav}/vins_fusion/imu_propagate`，类型 `nav_msgs/Odometry`。

## 当前实现目标

- [x] 新增 `groundctrl` ROS 包，放在 `core/src/groundctrl`。
- [x] 提供 `groundctrl.yaml` 统一管理 ground 和三台 UAV 的 IP、SSH host、ROS master、话题名。
- [x] 提供 ground 侧 `fkie_master_discovery`/`fkie_master_sync` launch，用于把三机遥测和 RViz 话题同步到地面站。
- [x] 提供 `groundctl.py`：
  - [x] 三机 SSH 状态检查。
  - [x] 远程启动 `roscore`、`core/shfiles/basectrl.sh`、`swarm_position_bridge`。
  - [x] 远程执行普通 shell/ROS topic 命令。
  - [x] 起飞/降落。
  - [x] EGO 单点、多点定时、多点到达判定飞行。
  - [x] PX4CTRL 单点、多点定时、多点到达判定位置控制。
  - [x] PX4CTRL 速度控制。
- [x] 提供 ground 状态监控节点，订阅三机里程计并发布地面站路径。

## 下一步现场验证

- [ ] 在 ground 上启动本地 `roscore`。
- [ ] 在 ground 上启动 `roslaunch groundctrl ground_sync.launch start_rviz:=true`。
- [ ] 逐台 UAV 验证 `groundctl status --uav uavX` 能看到 ROS master。
- [ ] 逐台 UAV 验证 `groundctl takeoff --uav uavX`、`groundctl land --uav uavX`。
- [ ] 逐台 UAV 验证 EGO 单点和到达判定任务。
- [ ] 逐台 UAV 验证 PX4CTRL 位置控制和速度控制。
- [ ] 按现场实际 RViz 话题修正 `ground_sync.launch` 中的显示话题清单。
