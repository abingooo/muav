# Ground Local Dependencies

为了让 ground 本机完整编译 `core`，已补齐这些系统/ROS 依赖：

```bash
sudo apt-get install -y \
  libceres-dev \
  ros-noetic-ddynamic-reconfigure \
  ros-noetic-librealsense2 \
  ros-noetic-mavros \
  ros-noetic-mavros-extras \
  geographiclib-tools
```

MAVROS 运行时 GeographicLib 数据集也已安装：

```text
/usr/share/GeographicLib/geoids/egm96-5.pgm
/usr/share/GeographicLib/gravity/egm96.egm
/usr/share/GeographicLib/magnetic/emm2015.wmm
```

官方脚本 `/opt/ros/noetic/lib/mavros/install_geographiclib_datasets.sh` 在当前网络下卡在 SourceForge 自动镜像选择，实际安装时改用 `curl -L` 下载并解包：

```text
https://downloads.sourceforge.net/project/geographiclib/geoids-distrib/egm96-5.tar.bz2
https://downloads.sourceforge.net/project/geographiclib/gravity-distrib/egm96.tar.bz2
https://downloads.sourceforge.net/project/geographiclib/magnetic-distrib/emm2015.tar.bz2
```
