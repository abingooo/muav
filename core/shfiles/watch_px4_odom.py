#!/usr/bin/env python3
import argparse
import math
import sys
import time

import rospy
import rostopic
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


def yaw_from_quat(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def pose_from_msg(msg):
    if isinstance(msg, PoseStamped):
        return msg.pose
    if isinstance(msg, Odometry):
        return msg.pose.pose
    return None


class Px4OdomWatcher:
    def __init__(self, args):
        self.args = args
        self.data = None
        self.zero_pose = None
        self.msg_class = None
        self.resolved_topic = None

    def start(self):
        rospy.loginfo("Waiting for topic %s", self.args.topic)
        msg_class, resolved_topic, _msg_eval = rostopic.get_topic_class(self.args.topic, blocking=True)
        if msg_class not in (PoseStamped, Odometry):
            rospy.logerr(
                "Unsupported topic type on %s: %s. Expected geometry_msgs/PoseStamped or nav_msgs/Odometry.",
                resolved_topic,
                msg_class._type,
            )
            return False

        self.msg_class = msg_class
        self.resolved_topic = resolved_topic
        rospy.Subscriber(resolved_topic, msg_class, self.cb, queue_size=1)
        return True

    def cb(self, msg):
        self.data = (msg, rospy.Time.now())

    def print_once(self):
        zero_mode = "display-zero" if self.args.zero else "raw"
        lines = [
            "\033[2J\033[H",
            f"watch_px4_odom  {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"topic   : {self.resolved_topic or self.args.topic}",
            f"type    : {self.msg_class._type if self.msg_class else 'waiting'}",
            f"mode    : {zero_mode}",
            "",
            "+------+------------+------------+------------+------------+----------+",
            "| src  |       x(m) |       y(m) |       z(m) |   yaw(deg) |   age(s) |",
            "+------+------------+------------+------------+------------+----------+",
        ]

        if self.data is None:
            lines.append(
                f"| PX4  | {'waiting':>10} | {'':>10} | {'':>10} | {'':>10} | {'':>8} |"
            )
            frame_id = "waiting"
            lines.append("+------+------------+------------+------------+------------+----------+")
        else:
            msg, stamp = self.data
            pose = pose_from_msg(msg)
            p = pose.position
            yaw = yaw_from_quat(pose.orientation)
            if self.args.zero and self.zero_pose is None:
                self.zero_pose = (p.x, p.y, p.z, yaw)

            if self.zero_pose is None:
                x, y, z, yaw_display = p.x, p.y, p.z, yaw
            else:
                x0, y0, z0, yaw0 = self.zero_pose
                x = p.x - x0
                y = p.y - y0
                z = p.z - z0
                yaw_display = math.atan2(math.sin(yaw - yaw0), math.cos(yaw - yaw0))

            yaw_deg = math.degrees(yaw_display)
            age = (rospy.Time.now() - stamp).to_sec()
            frame_id = msg.header.frame_id if msg.header.frame_id else "-"
            lines.append(
                f"| PX4  | {x:10.3f} | {y:10.3f} | {z:10.3f} | {yaw_deg:10.1f} | {age:8.2f} |"
            )
            lines.append("+------+------------+------------+------------+------------+----------+")

            if isinstance(msg, Odometry):
                v = msg.twist.twist.linear
                lines.extend(
                    [
                        "",
                        "+------+------------+------------+------------+",
                        "| src  |    vx(m/s) |    vy(m/s) |    vz(m/s) |",
                        "+------+------------+------------+------------+",
                        f"| PX4  | {v.x:10.3f} | {v.y:10.3f} | {v.z:10.3f} |",
                        "+------+------------+------------+------------+",
                    ]
                )

        lines.extend(
            [
                "",
                f"frame_id: {frame_id}",
                "zero: display offset only; PX4 EKF/local origin is not modified.",
                "Ctrl-C to stop.",
            ]
        )

        print("\n".join(lines))
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Watch PX4 local odometry/local position in one screen.")
    parser.add_argument("--uav", default="uav1", help="UAV namespace, default: uav1")
    parser.add_argument("--rate", type=float, default=5.0, help="Print rate in Hz, default: 5")
    parser.add_argument("--topic", default=None, help="Default: /<uav>/mavros/local_position/pose")
    parser.add_argument("--zero", action="store_true", help="Display pose relative to the first received sample")
    args = parser.parse_args()

    ns = args.uav.strip("/")
    if args.topic is None:
        args.topic = f"/{ns}/mavros/local_position/pose"

    rospy.init_node("watch_px4_odom", anonymous=True)
    watcher = Px4OdomWatcher(args)
    if not watcher.start():
        sys.exit(1)

    rate = rospy.Rate(args.rate)
    while not rospy.is_shutdown():
        watcher.print_once()
        rate.sleep()


if __name__ == "__main__":
    main()
