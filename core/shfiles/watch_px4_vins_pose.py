#!/usr/bin/env python3
import argparse
import math
import sys
import time

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


def yaw_from_quat(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def wrap_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def pose_values(data):
    if data is None:
        return None

    msg, stamp = data
    if isinstance(msg, PoseStamped):
        pose = msg.pose
    else:
        pose = msg.pose.pose

    p = pose.position
    yaw_deg = math.degrees(yaw_from_quat(pose.orientation))
    age = (rospy.Time.now() - stamp).to_sec()
    return p.x, p.y, p.z, yaw_deg, age


def pose_row(name, data):
    values = pose_values(data)
    if values is None:
        return (
            f"| {name:<8} | {'waiting':>10} | {'':>10} | {'':>10} | "
            f"{'':>10} | {'':>8} | {'':>10} | {'WAIT':<6} |"
        )

    x, y, z, yaw_deg, age = values
    return (
        f"| {name:<8} | {x:10.3f} | {y:10.3f} | {z:10.3f} | "
        f"{yaw_deg:10.1f} | {age:8.2f} | {'':>10} | {'':<6} |"
    )


def extract_xyzyaw(data):
    msg, _stamp = data
    if isinstance(msg, PoseStamped):
        pose = msg.pose
    else:
        pose = msg.pose.pose

    p = pose.position
    return p.x, p.y, p.z, yaw_from_quat(pose.orientation)


class PoseWatcher:
    def __init__(self, args):
        self.args = args
        self.px4 = None
        self.vins = None

        rospy.Subscriber(args.px4_topic, PoseStamped, self.px4_cb, queue_size=1)
        rospy.Subscriber(args.vins_topic, Odometry, self.vins_cb, queue_size=1)

    def px4_cb(self, msg):
        self.px4 = (msg, rospy.Time.now())

    def vins_cb(self, msg):
        self.vins = (msg, rospy.Time.now())

    def print_once(self):
        lines = [
            "\033[2J\033[H",
            f"watch_px4_vins_pose  {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"PX4 topic : {self.args.px4_topic}",
            f"VINS topic: {self.args.vins_topic}",
            (
                "WARN threshold: "
                f"dxy>{self.args.warn_xy:.3f}m, "
                f"|dz|>{self.args.warn_z:.3f}m, "
                f"|dyaw|>{self.args.warn_yaw_deg:.1f}deg"
            ),
            "",
            "+----------+------------+------------+------------+------------+----------+------------+--------+",
            "| row      |       x/dx |       y/dy |       z/dz |  yaw/dyaw |   age(s) |     dxy(m) | status |",
            "+----------+------------+------------+------------+------------+----------+------------+--------+",
            pose_row("PX4", self.px4),
            pose_row("VINS", self.vins),
        ]

        if self.px4 is not None and self.vins is not None:
            px4_x, px4_y, px4_z, px4_yaw = extract_xyzyaw(self.px4)
            vins_x, vins_y, vins_z, vins_yaw = extract_xyzyaw(self.vins)
            dx = px4_x - vins_x
            dy = px4_y - vins_y
            dz = px4_z - vins_z
            dyaw = math.degrees(wrap_pi(px4_yaw - vins_yaw))
            dxy = math.hypot(dx, dy)
            mark = "OK"
            if dxy > self.args.warn_xy or abs(dz) > self.args.warn_z or abs(dyaw) > self.args.warn_yaw_deg:
                mark = "WARN"

            lines.append(
                f"| {'PX4-VINS':<8} | {dx:10.3f} | {dy:10.3f} | {dz:10.3f} | "
                f"{dyaw:10.1f} | {'':>8} | {dxy:10.3f} | {mark:<6} |"
            )
        else:
            missing = []
            if self.px4 is None:
                missing.append("PX4")
            if self.vins is None:
                missing.append("VINS")
            lines.append(
                f"| {'PX4-VINS':<8} | {'waiting':>10} | {','.join(missing):>10} | {'':>10} | "
                f"{'':>10} | {'':>8} | {'':>10} | {'WAIT':<6} |"
            )

        lines.extend(
            [
                "+----------+------------+------------+------------+------------+----------+------------+--------+",
                "",
                "Ctrl-C to stop.",
            ]
        )

        print("\n".join(lines))
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Watch PX4 local pose and VINS odometry side by side.")
    parser.add_argument("--uav", default="uav1", help="UAV namespace, default: uav1")
    parser.add_argument("--rate", type=float, default=5.0, help="Print rate in Hz, default: 5")
    parser.add_argument("--warn-xy", type=float, default=0.5, help="Warn when xy diff is larger than this, meters")
    parser.add_argument("--warn-z", type=float, default=0.3, help="Warn when z diff is larger than this, meters")
    parser.add_argument("--warn-yaw-deg", type=float, default=30.0, help="Warn when yaw diff is larger than this, degrees")
    parser.add_argument("--px4-topic", default=None)
    parser.add_argument("--vins-topic", default=None)
    args = parser.parse_args()

    ns = args.uav.strip("/")
    if args.px4_topic is None:
        args.px4_topic = f"/{ns}/mavros/local_position/pose"
    if args.vins_topic is None:
        args.vins_topic = f"/{ns}/vins_fusion/imu_propagate"

    rospy.init_node("watch_px4_vins_pose", anonymous=True)
    watcher = PoseWatcher(args)

    rate = rospy.Rate(args.rate)
    while not rospy.is_shutdown():
        watcher.print_once()
        rate.sleep()


if __name__ == "__main__":
    main()
