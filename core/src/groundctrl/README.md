# groundctrl

`groundctrl` 是 MUAV 地面站包，面向 `ground -> uav0/uav1/uav2` 的通信、显示、任务下发和远程启动。

## 基本模型

- 地面站显示：ground 本机运行 `roscore` 和 `ground_sync.launch`，通过 `fkie_master_discovery`/`fkie_master_sync` 同步三台 UAV 的遥测、EGO RViz 话题和调试话题。
- 任务下发：`groundctl.py` 直接连接目标 UAV 的 ROS master，发布到目标 UAV 的真实话题，不依赖多 master 桥是否同步命令话题。
- 远程启动：`groundctl.py start ...` 通过 SSH 在 UAV 上执行 `roscore`、`core/shfiles/basectrl.sh`、`swarm_position_bridge`。

## 编译

```bash
cd ~/Desktop/uav_project/muav/core
catkin_make --pkg quadrotor_msgs groundctrl
source devel/setup.bash
```

ground 本机完整编译 `core` 需要的系统依赖记录在 `docs/dependencies.md`。

## 地面站显示

```bash
roscore
```

另开终端：

```bash
cd ~/Desktop/uav_project/muav
./core/shfiles/groundsync.sh sync start_rviz:=true
```

`groundsync.sh` 会从 `groundctrl.yaml` 读取 ground 的 IP，并固定导出
`ROS_MASTER_URI=http://20.0.0.172:11311`、`ROS_IP=20.0.0.172`。不要直接继承
`ROS_MASTER_URI=http://localhost:11311` 去启动同步链路，否则地面站可能只能看到话题名，
但收不到跨主机消息。

监控节点会订阅：

- `/uav0/vins_fusion/imu_propagate`
- `/uav1/vins_fusion/imu_propagate`
- `/uav2/vins_fusion/imu_propagate`

并发布 RViz 友好的路径：

- `/groundctrl/uav0/odom_path`
- `/groundctrl/uav1/odom_path`
- `/groundctrl/uav2/odom_path`

## 常用控制命令

```bash
rosrun groundctrl groundctl.py status --uav all
rosrun groundctrl groundctl.py start roscore --uav all
rosrun groundctrl groundctl.py start core --uav uav0
rosrun groundctrl groundctl.py start core --uav uav0 --no-ego
rosrun groundctrl groundctl.py start vins-sync --uav all
rosrun groundctrl groundctl.py stop core --uav all
```

`start` 默认会在本地等待并输出启动反馈：

```text
[groundctl] uav1 core feedback: READY|PARTIAL|FAILED
  [OK] roscore reachable http://20.0.0.187:11311
  [OK] required nodes 7/7
  [OK] required topics 5/5
```

含义：

- `READY`：远端命令已执行，ROS master 可达，核心节点/话题存在，并且关键数据话题能收到消息。
- `PARTIAL`：节点或话题部分起来了，但相机、VINS、MAVROS 或里程计数据还没完全 ready。
- `FAILED`：地面站无法连到目标 ROS master，或启动/SSH 失败。

只检查不启动：

```bash
rosrun groundctrl groundctl.py check core --uav uav1
rosrun groundctrl groundctl.py check core --uav uav1 --no-ego
```

只发起远程启动、不等待反馈：

```bash
rosrun groundctrl groundctl.py start core --uav uav1 --no-wait-ready
rosrun groundctrl groundctl.py start core --uav uav1 --no-ego --no-wait-ready
```

脚本里需要失败时返回非 0，可加：

```bash
rosrun groundctrl groundctl.py start core --uav uav1 --strict-ready
```

关闭远端 ROS 进程：

```bash
rosrun groundctrl groundctl.py stop core --uav all
rosrun groundctrl groundctl.py stop core --uav all --force
rosrun groundctrl groundctl.py stop all --uav all
```

`stop core` 会先尝试 `rosnode kill -a`，再清理 `basectrl.sh`、`roslaunch`、`roscore/rosmaster`、RealSense、VINS、MAVROS、PX4CTRL、EGO 和 vins-sync 相关进程。`--force` 只在普通关闭仍有残留时使用。

起飞/降落：

```bash
rosrun groundctrl groundctl.py takeoff --uav uav0
rosrun groundctrl groundctl.py land --uav uav0
```

EGO 单点、多点定时、多点到达判定：

```bash
rosrun groundctrl groundctl.py ego single --uav uav0 1.0,0.0,0.6
rosrun groundctrl groundctl.py ego timed --uav uav0 --interval 5 1,0,0.6 2,0,0.6
rosrun groundctrl groundctl.py ego reached --uav uav0 --tolerance 0.3 1,0,0.6 2,0,0.6
```

PX4CTRL 位置/速度控制：

```bash
rosrun groundctrl groundctl.py px4 single --uav uav0 --mode position --duration 5 1,0,0.6
rosrun groundctrl groundctl.py px4 timed --uav uav0 --interval 5 1,0,0.6 2,0,0.6
rosrun groundctrl groundctl.py px4 reached --uav uav0 --tolerance 0.25 1,0,0.6 2,0,0.6
rosrun groundctrl groundctl.py px4 single --uav uav0 --mode velocity --velocity 0.2,0,0 --duration 3
```

多机并发时使用 `--uav all` 或逗号列表：

```bash
rosrun groundctrl groundctl.py takeoff --uav uav0,uav1,uav2
rosrun groundctrl groundctl.py ego reached --uav all 1,0,0.6
```

## 话题与 SSH

配置集中在：

```text
core/src/groundctrl/config/groundctrl.yaml
```

远程 ROS topic 示例：

```bash
rosrun groundctrl groundctl.py topic --uav uav0 list
rosrun groundctrl groundctl.py topic --uav uav0 echo /uav0/vins_fusion/imu_propagate -n 1
rosrun groundctrl groundctl.py remote --uav uav0 --ros-env rostopic hz /uav0/vins_fusion/imu_propagate
```

如果现场实际话题没有 `/uavX` 前缀，先改 `groundctrl.yaml` 的 `topics` 区，再改 `ground_sync.launch` 的同步列表。
