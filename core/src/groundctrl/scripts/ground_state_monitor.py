#!/usr/bin/env python3

import os
from collections import OrderedDict

import rospy
import yaml
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path


def load_config(path):
    if not path:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "groundctrl.yaml",
        )
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def topic(config, name, uav_name):
    uav_id = config["uavs"][uav_name]["id"]
    return config["topics"][name].format(uav=uav_name, id=uav_id)


class GroundStateMonitor:
    def __init__(self):
        config_file = rospy.get_param("~config_file", "")
        self.config = load_config(config_file)
        self.max_path_len = int(rospy.get_param("~max_path_len", 3000))
        self.last_odom = OrderedDict()
        self.paths = {}
        self.path_pubs = {}
        self.odom_subs = []

        for uav_name in sorted(self.config["uavs"], key=lambda name: self.config["uavs"][name]["id"]):
            self.paths[uav_name] = Path()
            self.paths[uav_name].header.frame_id = "world"
            self.path_pubs[uav_name] = rospy.Publisher(
                f"/groundctrl/{uav_name}/odom_path", Path, queue_size=1, latch=True
            )
            self.odom_subs.append(
                rospy.Subscriber(
                    topic(self.config, "odom", uav_name),
                    Odometry,
                    self._make_odom_cb(uav_name),
                    queue_size=20,
                )
            )

        self.summary_timer = rospy.Timer(rospy.Duration(2.0), self._summary_cb)
        rospy.loginfo("ground_state_monitor is tracking: %s", ", ".join(self.config["uavs"].keys()))

    def _make_odom_cb(self, uav_name):
        def callback(msg):
            self.last_odom[uav_name] = (rospy.Time.now(), msg)
            pose = PoseStamped()
            pose.header = msg.header
            if not pose.header.frame_id:
                pose.header.frame_id = "world"
            pose.pose = msg.pose.pose

            path = self.paths[uav_name]
            path.header.stamp = pose.header.stamp
            path.header.frame_id = pose.header.frame_id
            path.poses.append(pose)
            if len(path.poses) > self.max_path_len:
                path.poses = path.poses[-self.max_path_len :]
            self.path_pubs[uav_name].publish(path)

        return callback

    def _summary_cb(self, _event):
        now = rospy.Time.now()
        parts = []
        for uav_name in sorted(self.config["uavs"], key=lambda name: self.config["uavs"][name]["id"]):
            sample = self.last_odom.get(uav_name)
            if sample is None:
                parts.append(f"{uav_name}: no odom")
                continue
            stamp, msg = sample
            age = (now - stamp).to_sec()
            p = msg.pose.pose.position
            parts.append(f"{uav_name}: ({p.x:.2f}, {p.y:.2f}, {p.z:.2f}) age={age:.1f}s")
        rospy.loginfo("ground odom: %s", " | ".join(parts))


def main():
    rospy.init_node("ground_state_monitor")
    GroundStateMonitor()
    rospy.spin()


if __name__ == "__main__":
    main()
