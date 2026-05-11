# adv Standalone ROS Workspace

This directory is a standalone catkin workspace next to `core`. It contains the
`adv` ROS package and a minimal local `quadrotor_msgs` package used by the ADV
adapter, so it can build without sourcing `core`.

## Layout

```text
adv/
  src/
    adv/
      CMakeLists.txt
      package.xml
      setup.py
      scripts/
        adv_ros_adapter
        adv_ros_e2e_smoke_test
        adv_ros_e2e_speed_test
      adv_module/
    quadrotor_msgs/
      msg/PositionCommand.msg
```

## Core Interface

Use structured state input when you have positions and velocities:

```python
from adv_module import InferenceEngine, InferenceSnapshot, VehicleState

engine = InferenceEngine()

snapshot = InferenceSnapshot(
    defenders=[
        VehicleState(position=[1.0, 1.2, 1.0], velocity=[0.0, 0.0, 0.0]),
        VehicleState(position=[-1.4, 0.8, 1.0], velocity=[0.0, 0.0, 0.0]),
        VehicleState(position=[0.6, -1.5, 1.0], velocity=[0.0, 0.0, 0.0]),
        VehicleState(position=[-0.9, -1.0, 1.0], velocity=[0.0, 0.0, 0.0]),
    ],
    enemy=VehicleState(position=[28.0, 12.0, 1.0], velocity=[-4.0, -1.2, 0.0]),
    step_count=0,
)

result = engine.predict(snapshot)
for command in result.commands:
    print(command.name, command.role_name, command.velocity_xyz)
```

The default parameters live in `src/adv/adv_module/inference_config.yaml`. That file
uses YAML so comments can be kept next to the parameters. To run with another
config file:

```python
engine = InferenceEngine(config_path="path/to/inference_config.yaml")
```

Or use prebuilt 25D observations:

```python
actions = engine.predict_from_observations([obs_d1, obs_d2, obs_d3, obs_d4])
```

Run from the `adv` folder with:

```bash
PYTHONPATH=src/adv python3 your_script.py
```

## ROS Adapter

`src/adv/adv_module/ros_adapter.py` is the ROS entry point. It reads
`src/adv/adv_module/model_config.yaml`, subscribes to the configured five
`vins_position` topics, builds a four-defender plus one-enemy inference
snapshot, and publishes one selected defender command as
`quadrotor_msgs/PositionCommand`.

Default input mapping:

```text
/uav0/vins_position -> defender_0
/uav1/vins_position -> defender_1
/uav2/vins_position -> defender_2
/uav3/vins_position -> defender_3
/uav5/vins_position -> enemy
```

Build and run it without sourcing `core`:

```bash
cd /home/uav/Desktop/uav_project/muav/adv
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
rosrun adv adv_ros_adapter
```

Or launch it with:

```bash
UAV_NAME=${UAV_NAME:-uav} roslaunch adv adv.launch enemy_uav:=uav2
```

`adv` publishes the original `quadrotor_msgs/PositionCommand` and, when
`publish_plan_point` is true, also publishes a converted
`geometry_msgs/PoseStamped` plan point for core.

In launch mode, `UAV_NAME` is the UAV running this ADV instance.
`enemy_uav` selects which real UAV is the enemy input. Defender roles are stable
slots: `uav0 -> defender_0`, `uav1 -> defender_1`, `uav2 -> defender_2`, and
`uav3 -> defender_3`. If a defender slot is occupied by `enemy_uav`, the next
extra UAV in `fleet_uavs` backs that slot. For example:

```bash
UAV_NAME=uav4 roslaunch adv adv.launch enemy_uav:=uav2 fleet_uavs:=uav0,uav1,uav2,uav3,uav4
```

maps `enemy <- /uav2/vins_position` and `defender_2 <- /uav4/vins_position`;
the ADV instance on `uav4` automatically publishes the `defender_2` command to
`/uav4/position_cmd`.

### Game End Handling

The adapter can stop the match from `model_config.yaml` `game_end` settings:

- defender win: enemy is within `capture_distance_m` of any defender in 3D for
  `hold_duration_sec`
- enemy win: enemy is within `asset_distance_m` horizontal distance of the
  critical asset for `hold_duration_sec`
- experiment failed: any received aircraft state leaves the configured world
  bounds

The critical asset position is loaded from ADV `inference_config.yaml`
`inference.origin`. Default world bounds are loaded from ADV
`inference_config.yaml` `safety.bounds`. MPC keeps its own independent
`mpc_config.yaml` `environment.origin`.
When a terminal state is triggered, ADV publishes one current-position
`PositionCommand` to every mapped UAV command topic and then stops publishing
normal strategy commands.

Or pass a different model config through a private ROS parameter:

```bash
rosrun adv adv_ros_adapter _model_config:=/path/to/model_config.yaml
```

End-to-end smoke test, after `roscore` and the adapter are running:

```bash
cd /home/uav/Desktop/uav_project/muav/adv
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rosrun adv adv_ros_e2e_smoke_test
```

The smoke test publishes fake `vins_position` inputs from `model_config.yaml`
and waits for a valid `PositionCommand` on the configured output topic.

End-to-end speed limit test, after `roscore` is running. Do not start a separate
adapter for this test; the script starts an in-process adapter with a temporary
high `publish_rate_hz`.

```bash
cd /home/uav/Desktop/uav_project/muav/adv
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rosrun adv adv_ros_e2e_speed_test --duration 10
```

The speed limit test publishes fake inputs, temporarily sets adapter
`publish_rate_hz` to `--adapter-rate` (`1000` by default), and measures the
actual output throughput limit on the configured `PositionCommand` topic.

```bash
rosrun adv adv_ros_e2e_speed_test --adapter-rate 2000 --input-rate 300 --duration 15
```

To measure an already-running adapter instead:

```bash
rosrun adv adv_ros_e2e_speed_test --no-start-adapter
```

### Coordinate Transforms

Each input topic is assumed to be in that UAV's own local motion frame. Configure
`coordinate_transforms` in `model_config.yaml` to convert each local frame into
the ADV world frame before inference:

```text
world_position = Rz(yaw_deg) * local_position + translation
world_velocity = Rz(yaw_deg) * local_velocity
```

`translation` is the UAV local origin in the ADV world frame. `yaw_deg` is the
heading of the UAV local x axis relative to the ADV world x axis. By default all
transforms are identity.

Adapter output uses `adapter.output_frame`:

```text
local -> transform the ADV world command back into output_role's local frame
world -> publish the ADV world command directly
```

## Dependencies

Install the runtime dependencies listed in `requirements.txt`. The dependency
set uses CPU-only PyTorch by default, so it is suitable for machines without a
GPU.
