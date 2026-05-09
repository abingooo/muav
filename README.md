# MUAV

MUAV 是一个多无人机 ROS 工程集合，用于运行 `core` 飞控/感知/规划链路，以及相互独立的 `adv`、`mpc` 策略包。

## 项目结构

```text
muav/
  core/          # 飞控、感知、规划和运行脚本
  adv/           # 独立 catkin 工作区，ADV 策略
  mpc/           # 独立 catkin 工作区，MPC 策略
  autouavenv.sh  # 自动写入 ~/.bashrc 的环境初始化脚本
  cmd.md         # 现场运行命令速查
```

`adv` 和 `mpc` 与 `core` 同层级，都是独立 catkin 工作区；它们不 source `core/devel`，也不依赖 `core` 的编译环境。

## 环境变量

工程统一使用 `UAV_NAME` 表示当前机器/无人机名称，例如：

```bash
UAV_NAME=uav0
UAV_NAME=uav1
UAV_NAME=uav2
```

默认值是：

```bash
UAV_NAME=uav
```

项目中不再使用旧的 ROS namespace 环境变量。

## 自动环境脚本

编译完成后执行一次：

```bash
cd /home/uav/Desktop/uav_project/muav
./autouavenv.sh
```

脚本会自动扫描当前工程下所有：

```text
*/devel/setup.bash
```

然后写入 `~/.bashrc` 的受控标记块中。重复执行不会重复追加，会替换旧块。

如果执行脚本时当前环境中已经有合法的 `UAV_NAME`，例如：

```bash
UAV_NAME=uav3 ./autouavenv.sh
```

则 `.bashrc` 中会写入：

```bash
export UAV_NAME=uav3
```

如果没有设置，或设置值不是 `uav0` 到 `uav4`，则写入：

```bash
export UAV_NAME=uav
```

## 编译

三个工作区分别编译：

```bash
cd /home/uav/Desktop/uav_project/muav/core
catkin_make

cd /home/uav/Desktop/uav_project/muav/adv
catkin_make

cd /home/uav/Desktop/uav_project/muav/mpc
catkin_make
```

## 运行 Core

启动基础控制链路：

```bash
cd /home/uav/Desktop/uav_project/muav
UAV_NAME=uav0 ./core/shfiles/basectrl.sh
```

起飞：

```bash
UAV_NAME=uav0 ./core/shfiles/takeoff.sh
```

降落：

```bash
UAV_NAME=uav0 ./core/shfiles/land.sh
```

## 运行 MPC

`mpc` 的规则是：当前 `UAV_NAME` 对应的无人机就是 MPC 内部的 `enemy`。

例如部署到 `uav2`：

```bash
cd /home/uav/Desktop/uav_project/muav/mpc
UAV_NAME=uav2 roslaunch mpc mpc.launch
```

此时输出：

```text
/uav2/position_cmd
/uav2/toplan/single_plan_point
```

## 运行 ADV

`adv` 的 defender 槽位是稳定映射：

```text
uav0 -> defender_0
uav1 -> defender_1
uav2 -> defender_2
uav3 -> defender_3
```

敌机通过 `enemy_uav` 指定。

例如当前在 `uav0` 上运行 ADV，敌机是 `uav2`：

```bash
cd /home/uav/Desktop/uav_project/muav/adv
UAV_NAME=uav0 roslaunch adv adv.launch enemy_uav:=uav2
```

如果敌机占用了某个 defender 槽位，则由后备无人机顶替。默认机队是：

```text
uav0,uav1,uav2,uav3,uav4
```

例如敌机是 `uav2`，`uav4` 顶替 `defender_2`：

```bash
UAV_NAME=uav4 roslaunch adv adv.launch enemy_uav:=uav2
```

此时映射为：

```text
enemy      <- /uav2/vins_position
defender_0 <- /uav0/vins_position
defender_1 <- /uav1/vins_position
defender_2 <- /uav4/vins_position
defender_3 <- /uav3/vins_position
```

并输出：

```text
/uav4/position_cmd
/uav4/toplan/single_plan_point
```

## 常用命令

现场运行命令见：

```text
cmd.md
```
