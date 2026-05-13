# Groundctrl Network Notes

## ROS 1 多主机条件

每台机器都需要让其它机器能反向连接到自己的 ROS 节点 XMLRPC/TCPROS 地址。

ground:

```bash
export ROS_MASTER_URI=http://20.0.0.172:11311
export ROS_IP=20.0.0.172
```

uav0:

```bash
export ROS_MASTER_URI=http://20.0.0.188:11311
export ROS_IP=20.0.0.188
```

uav1:

```bash
export ROS_MASTER_URI=http://20.0.0.187:11311
export ROS_IP=20.0.0.187
```

uav2:

```bash
export ROS_MASTER_URI=http://20.0.0.208:11311
export ROS_IP=20.0.0.208
```

`groundctl.py` 在发布控制命令时会自动把本进程设置为：

```bash
ROS_MASTER_URI=http://目标UAV_IP:11311
ROS_IP=20.0.0.172
```

这样目标 UAV 的订阅者能直接连回 ground 的发布者。

## Namespace

`groundctrl.yaml` 默认使用 `/uav0`、`/uav1`、`/uav2` 话题前缀。`groundctl.py start core` 默认在远端设置：

```bash
ROS_NAMESPACE=/uavX
UAV_NAME=uavX
```

如果现场已经用其它方式做 namespace，可以加：

```bash
rosrun groundctrl groundctl.py start core --uav uav0 --no-namespace
```
