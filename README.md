# MUAV

Multi-UAV ROS workspace collection for running the core flight stack together
with independent ADV and MPC strategy packages.

## Layout

```text
muav/
  core/          # flight control, perception, planning, and shell launch helpers
  adv/           # independent catkin workspace for ADV strategy
  mpc/           # independent catkin workspace for MPC strategy
  autouavenv.sh  # writes the local ROS environment block into ~/.bashrc
  cmd.md         # compact field-run command reference
```

`adv` and `mpc` are independent catkin workspaces at the same level as `core`.
They do not source or depend on `core/devel`.

## Environment

Run this once after building the workspaces:

```bash
cd /home/uav/Desktop/uav_project/muav
./autouavenv.sh
```

The script scans all `*/devel/setup.bash` files under this repository and writes
a managed block to `~/.bashrc`. It also writes `UAV_NAME`:

```bash
export UAV_NAME=uav
```

If `UAV_NAME` is already set to `uav0` through `uav4` when the script runs, that
value is persisted instead.

## Build

Build each catkin workspace separately:

```bash
cd /home/uav/Desktop/uav_project/muav/core
catkin_make

cd /home/uav/Desktop/uav_project/muav/adv
catkin_make

cd /home/uav/Desktop/uav_project/muav/mpc
catkin_make
```

## Run Core

```bash
cd /home/uav/Desktop/uav_project/muav
UAV_NAME=uav0 ./core/shfiles/basectrl.sh
```

Take off:

```bash
UAV_NAME=uav0 ./core/shfiles/takeoff.sh
```

## Run MPC

MPC treats the current `UAV_NAME` as the internal `enemy` role.

```bash
cd /home/uav/Desktop/uav_project/muav/mpc
UAV_NAME=uav2 roslaunch mpc mpc.launch
```

For `UAV_NAME=uav2`, MPC publishes:

```text
/uav2/position_cmd
/uav2/toplan/single_plan_point
```

## Run ADV

ADV uses stable defender slots:

```text
uav0 -> defender_0
uav1 -> defender_1
uav2 -> defender_2
uav3 -> defender_3
```

The enemy UAV is selected explicitly:

```bash
cd /home/uav/Desktop/uav_project/muav/adv
UAV_NAME=uav0 roslaunch adv adv.launch enemy_uav:=uav2
```

If the enemy occupies a defender slot, the next extra UAV in `fleet_uavs`
backs that slot. With the default fleet `uav0,uav1,uav2,uav3,uav4`, this command:

```bash
UAV_NAME=uav4 roslaunch adv adv.launch enemy_uav:=uav2
```

maps `uav4` to `defender_2` and publishes:

```text
/uav4/position_cmd
/uav4/toplan/single_plan_point
```

## Command Reference

See [`cmd.md`](cmd.md) for the compact run commands used during field testing.
