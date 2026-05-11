# MPC Local Planner

This folder is now structured like `adv`: local strategy core, YAML configs,
ROS adapter, and end-to-end tests. The old HTTP server path has been removed.

## Files

```text
mpc/
  src/
    CMakeLists.txt
    mpc/
      CMakeLists.txt
      package.xml
      setup.py
      scripts/
        mpc_ros_adapter
      launch/
        mpc.launch
      mpc_module/
        enemy_strategy.py
        mpc_config.yaml
        model_config.yaml
        mpc_engine.py
        ros_adapter.py
```

## Core Usage

```python
from mpc_module import MpcEngine, MpcSnapshot, VehicleState

engine = MpcEngine()
result = engine.plan(MpcSnapshot(
    enemy=VehicleState(position=[5.0, 0.0, 1.0], velocity=[0.0, 0.0, 0.0]),
    defenders=[
        VehicleState(position=[1.0, 0.0, 1.0], velocity=[0.0, 0.0, 0.0]),
        VehicleState(position=[0.5, 1.0, 1.0], velocity=[0.0, 0.0, 0.0]),
        VehicleState(position=[0.5, -1.0, 1.0], velocity=[0.0, 0.0, 0.0]),
        VehicleState(position=[1.5, 0.0, 1.0], velocity=[0.0, 0.0, 0.0]),
    ],
    step_count=1,
))
print(result.predicted_position, result.velocity_xyz, result.debug)
```

## ROS Adapter

The adapter subscribes to the configured `vins_position` inputs, converts each
UAV local frame into the MPC world frame, runs the local MPC engine, then
publishes a native `quadrotor_msgs/PositionCommand`. When `publish_plan_point`
is enabled, it also publishes a converted `geometry_msgs/PoseStamped` plan point
for core.

```bash
cd /home/uav/Desktop/uav_project/muav/mpc
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
rosrun mpc mpc_ros_adapter
```

Or launch it with:

```bash
cd /home/uav/Desktop/uav_project/muav/mpc
source /opt/ros/noetic/setup.bash
source devel/setup.bash
UAV_NAME=${UAV_NAME:-uav} roslaunch mpc mpc.launch
```

In launch mode, `UAV_NAME` is the MPC-controlled UAV. That UAV is mapped
to the internal `enemy` role, and the remaining names in `fleet_uavs` are mapped
to `defender_0..defender_3` in order. For example:

```bash
UAV_NAME=uav2 roslaunch mpc mpc.launch fleet_uavs:=uav0,uav1,uav2,uav3,uav4
```

This publishes the native command to `/uav2/position_cmd`, subscribes
`enemy <- /uav2/vins_position`, and maps the other fleet members to defender
inputs. The converted plan point is published to
`/${UAV_NAME}/toplan/single_plan_point`, matching core's
`geometry_msgs/PoseStamped` plan-point input.

### Game End Handling

The adapter can stop the match from `model_config.yaml` `game_end` settings:

- defender win: enemy is within `capture_distance_m` of any defender in 3D for
  `hold_duration_sec`
- enemy win: enemy is within `asset_distance_m` horizontal distance of the
  critical asset for `hold_duration_sec`
- experiment failed: any received aircraft state leaves the configured world
  bounds

The critical asset position is loaded from MPC `mpc_config.yaml`
`environment.origin`. Default world bounds are loaded from MPC
`mpc_config.yaml` `environment.x_min`, `x_max`, `y_min`, and `y_max`.
ADV keeps its own independent `inference_config.yaml` `inference.origin`.
When a terminal state is triggered, MPC publishes one current-position
`PositionCommand` to every mapped UAV command topic and then stops publishing
normal strategy commands.

## Tests

Run `roscore` first.

Smoke test:

```bash
cd /home/uav/Desktop/uav_project/muav
source core/devel/setup.bash
PYTHONPATH=mpc:$PYTHONPATH python3 -m mpc_module.ros_e2e_smoke_test
```

Speed-limit test:

```bash
PYTHONPATH=mpc:$PYTHONPATH python3 -m mpc_module.ros_e2e_speed_test --duration 10
```

The speed test starts an in-process adapter with a temporary high
`publish_rate_hz` and reports measured output throughput.
