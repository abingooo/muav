#!/usr/bin/env python3

import argparse
import math
import os
import shutil
import sys
import time
from collections import deque
from pathlib import Path

import rospy
import yaml
from nav_msgs.msg import Odometry


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_DIR = SCRIPT_PATH.parent.parent
DEFAULT_CONFIG = PACKAGE_DIR / "config" / "groundctrl.yaml"


def load_config(path):
    path = Path(path or DEFAULT_CONFIG).expanduser()
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ordered_uavs(config):
    return sorted(config["uavs"], key=lambda name: int(config["uavs"][name]["id"]))


def resolve_targets(config, target_spec):
    if target_spec in (None, "", "all"):
        return ordered_uavs(config)
    targets = [item.strip() for item in target_spec.split(",") if item.strip()]
    unknown = [name for name in targets if name not in config["uavs"]]
    if unknown:
        raise SystemExit(f"Unknown UAV target(s): {', '.join(unknown)}")
    return targets


def topic(config, topic_name, uav_name):
    uav_conf = config["uavs"][uav_name]
    return config["topics"][topic_name].format(uav=uav_name, id=uav_conf["id"])


def color(text, code, enabled):
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


def fmt_num(value, width=7, precision=2):
    if value is None or not math.isfinite(value):
        return "--".rjust(width)
    return f"{value:{width}.{precision}f}"


def fmt_age(value):
    if value is None or not math.isfinite(value):
        return "   --"
    return f"{value:5.1f}"


class OdomSlot:
    def __init__(self, uav_name, topic_name):
        self.uav_name = uav_name
        self.topic_name = topic_name
        self.last_msg = None
        self.last_wall = None
        self.samples = deque(maxlen=200)

    def update(self, msg):
        now = time.monotonic()
        self.last_msg = msg
        self.last_wall = now
        self.samples.append(now)

    def age(self, now):
        if self.last_wall is None:
            return None
        return now - self.last_wall

    def hz(self, now, window_s):
        cutoff = now - window_s
        while self.samples and self.samples[0] < cutoff:
            self.samples.popleft()
        if len(self.samples) < 2:
            return 0.0
        span = self.samples[-1] - self.samples[0]
        if span <= 0.0:
            return 0.0
        return (len(self.samples) - 1) / span

    def pose_values(self):
        if self.last_msg is None:
            return None
        pose = self.last_msg.pose.pose
        twist = self.last_msg.twist.twist
        return (
            pose.position.x,
            pose.position.y,
            pose.position.z,
            twist.linear.x,
            twist.linear.y,
            twist.linear.z,
        )


class OdomWatch:
    def __init__(self, args):
        self.args = args
        self.config = load_config(args.config)
        self.targets = resolve_targets(self.config, args.uav)
        self.slots = {
            uav_name: OdomSlot(uav_name, topic(self.config, "odom", uav_name))
            for uav_name in self.targets
        }
        self.subscribers = []

        rospy.init_node("ground_odom_watch", anonymous=True, disable_signals=True)
        for uav_name, slot in self.slots.items():
            self.subscribers.append(
                rospy.Subscriber(
                    slot.topic_name,
                    Odometry,
                    self._make_odom_cb(slot),
                    queue_size=20,
                )
            )

    def _make_odom_cb(self, slot):
        def callback(msg):
            slot.update(msg)

        return callback

    def state_for_age(self, age):
        if age is None:
            return "LOST"
        if age >= self.args.lost_age:
            return "LOST"
        if age >= self.args.warn_age:
            return "STALE"
        return "OK"

    def state_text(self, state):
        if state == "OK":
            return color(state.ljust(5), "32", self.args.color)
        if state == "STALE":
            return color(state.ljust(5), "33", self.args.color)
        return color(state.ljust(5), "31", self.args.color)

    def render(self):
        now = time.monotonic()
        wall = time.strftime("%H:%M:%S")
        width = shutil.get_terminal_size((120, 30)).columns
        master = os.environ.get("ROS_MASTER_URI", "")
        lines = [
            f"MUAV odom watch  target={self.args.uav}  master={master}  time={wall}",
            f"warn_age={self.args.warn_age:.1f}s  lost_age={self.args.lost_age:.1f}s  hz_window={self.args.hz_window:.1f}s",
            "",
            "UAV   STATE AGE(s)     HZ       X       Y       Z      VX      VY      VZ  TOPIC",
            "----  ----- ------ ------ ------- ------- ------- ------- ------- -------  -----",
        ]

        for uav_name in self.targets:
            slot = self.slots[uav_name]
            age = slot.age(now)
            state = self.state_for_age(age)
            values = slot.pose_values()
            hz = slot.hz(now, self.args.hz_window)
            if values is None:
                x = y = z = vx = vy = vz = None
            else:
                x, y, z, vx, vy, vz = values
            lines.append(
                f"{uav_name:<4}  {self.state_text(state)} {fmt_age(age)} "
                f"{hz:6.1f} {fmt_num(x)} {fmt_num(y)} {fmt_num(z)} "
                f"{fmt_num(vx)} {fmt_num(vy)} {fmt_num(vz)}  {slot.topic_name}"
            )

        lines.append("")
        lines.append("Ctrl-C to exit. OK: fresh odom, STALE: delayed odom, LOST: no recent odom.")
        clipped = [line[:width] if width > 0 else line for line in lines]
        return "\n".join(clipped)

    def run(self):
        interval = 1.0 / max(0.1, self.args.rate)
        if not self.args.once and sys.stdout.isatty():
            sys.stdout.write("\033[?25l")
            sys.stdout.flush()
        try:
            while not rospy.is_shutdown():
                if self.args.once or not sys.stdout.isatty():
                    print(self.render())
                else:
                    sys.stdout.write("\033[H\033[J")
                    sys.stdout.write(self.render())
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                if self.args.once:
                    break
                time.sleep(interval)
        finally:
            if not self.args.once and sys.stdout.isatty():
                sys.stdout.write("\033[?25h")
                sys.stdout.flush()


def build_parser():
    parser = argparse.ArgumentParser(description="Fixed-screen MUAV odometry dashboard")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="groundctrl.yaml path")
    parser.add_argument("--uav", default="all", help="Target UAV: uav0, uav1, uav2, comma list, or all")
    parser.add_argument("--rate", type=float, default=2.0, help="Screen refresh rate in Hz")
    parser.add_argument("--warn-age", type=float, default=1.0, help="Mark odom STALE after this many seconds")
    parser.add_argument("--lost-age", type=float, default=3.0, help="Mark odom LOST after this many seconds")
    parser.add_argument("--hz-window", type=float, default=3.0, help="Message-rate measurement window in seconds")
    parser.add_argument("--no-color", dest="color", action="store_false", help="Disable terminal colors")
    parser.add_argument("--once", action="store_true", help="Print one table and exit")
    parser.set_defaults(color=sys.stdout.isatty())
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    if args.lost_age < args.warn_age:
        raise SystemExit("--lost-age must be greater than or equal to --warn-age")
    OdomWatch(args).run()


if __name__ == "__main__":
    main()
