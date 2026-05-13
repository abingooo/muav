#!/usr/bin/env python3

import argparse
import math
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_DIR = SCRIPT_PATH.parent.parent
DEFAULT_CONFIG = PACKAGE_DIR / "config" / "groundctrl.yaml"
ROS_ACTIONS = {"takeoff", "land", "ego", "px4"}


def load_config(path):
    path = Path(path or DEFAULT_CONFIG).expanduser()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config


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


def parse_vector(text, label="vector"):
    clean = text.replace(",", " ")
    values = [part for part in clean.split() if part]
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"{label} must have 3 numbers, got: {text}")
    try:
        return tuple(float(value) for value in values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} contains a non-number: {text}") from exc


def parse_points(args, required=True):
    raw_items = []
    if getattr(args, "points_text", None):
        raw_items.extend(
            item.strip()
            for item in args.points_text.replace("|", ";").split(";")
            if item.strip()
        )
    raw_items.extend(getattr(args, "point_args", []) or [])
    points = [parse_vector(item, "point") for item in raw_items]
    if required and not points:
        raise SystemExit("At least one point is required, for example: 1,0,0.6")
    return points


def looks_like_vector_arg(text):
    try:
        parse_vector(text, "point")
        return True
    except argparse.ArgumentTypeError:
        return False


def parse_args_allowing_negative_vectors(parser):
    args, unknown = parser.parse_known_args()
    if not unknown:
        return args

    if hasattr(args, "point_args") and all(looks_like_vector_arg(item) for item in unknown):
        args.point_args.extend(unknown)
        return args

    parser.error(f"unrecognized arguments: {' '.join(unknown)}")


def yaw_rad_from_args(args):
    yaw_deg = getattr(args, "yaw_deg", None)
    if yaw_deg is None:
        return 0.0
    return float(yaw_deg) * math.pi / 180.0


def replace_or_append_uav_arg(argv, target):
    result = list(argv)
    for idx, item in enumerate(result):
        if item == "--uav" and idx + 1 < len(result):
            result[idx + 1] = target
            return result
        if item.startswith("--uav="):
            result[idx] = f"--uav={target}"
            return result
    result.extend(["--uav", target])
    return result


def fan_out_if_needed(config, args):
    targets = resolve_targets(config, getattr(args, "uav", None))
    if len(targets) <= 1 or args.command not in ROS_ACTIONS:
        return False

    processes = []
    for target in targets:
        child_argv = replace_or_append_uav_arg(sys.argv[1:], target)
        cmd = [sys.executable, str(SCRIPT_PATH)] + child_argv
        print(f"[groundctl] start {target}: {' '.join(shlex.quote(part) for part in cmd)}")
        processes.append((target, subprocess.Popen(cmd, env=os.environ.copy())))

    failed = 0
    for target, proc in processes:
        ret = proc.wait()
        if ret != 0:
            failed = ret if failed == 0 else failed
            print(f"[groundctl] {target} failed with exit code {ret}", file=sys.stderr)
    raise SystemExit(failed)


def configure_target_ros_env(config, uav_name):
    ground = config["ground"]
    uav_conf = config["uavs"][uav_name]
    os.environ["ROS_MASTER_URI"] = uav_conf["ros_master_uri"]
    os.environ["ROS_IP"] = str(ground["ip"])
    os.environ.pop("ROS_HOSTNAME", None)
    os.environ["UAV_NAME"] = uav_name


def wait_for_pub_connections(rospy, publisher, timeout_s):
    deadline = time.time() + max(0.0, timeout_s)
    while time.time() < deadline and not rospy.is_shutdown():
        if publisher.get_num_connections() > 0:
            return True
        rospy.sleep(0.05)
    return publisher.get_num_connections() > 0


def require_pub_connections(rospy, publisher, topic_name, timeout_s, purpose):
    if wait_for_pub_connections(rospy, publisher, timeout_s):
        count = publisher.get_num_connections()
        print(f"[groundctl] {purpose}: {count} subscriber(s) connected on {topic_name}")
        return
    raise SystemExit(
        f"[groundctl] {purpose}: no subscriber connected on {topic_name} "
        f"within {timeout_s:.1f}s; command would have no effect"
    )


class MissionObserver:
    def __init__(self, rospy, config, uav_name):
        from nav_msgs.msg import Odometry
        from std_msgs.msg import String

        self.rospy = rospy
        self.odom = None
        self.odom_stamp = rospy.Time(0)
        self.done_stamp = rospy.Time(0)
        self.done_text = ""
        self.odom_sub = rospy.Subscriber(
            topic(config, "odom", uav_name), Odometry, self._odom_cb, queue_size=20
        )
        self.done_sub = rospy.Subscriber(
            topic(config, "ego_done", uav_name), String, self._done_cb, queue_size=10
        )

    def _odom_cb(self, msg):
        self.odom = msg
        self.odom_stamp = self.rospy.Time.now()

    def _done_cb(self, msg):
        self.done_text = msg.data
        self.done_stamp = self.rospy.Time.now()

    def wait_odom(self, timeout_s):
        deadline = time.time() + timeout_s
        while self.odom is None and time.time() < deadline and not self.rospy.is_shutdown():
            self.rospy.sleep(0.05)
        return self.odom is not None

    def distance_to(self, point):
        if self.odom is None:
            return float("inf")
        pos = self.odom.pose.pose.position
        return math.sqrt(
            (pos.x - point[0]) ** 2 +
            (pos.y - point[1]) ** 2 +
            (pos.z - point[2]) ** 2
        )

    def wait_reached(self, point, start_stamp, tolerance, hold_s, timeout_s, use_done):
        deadline = time.time() + timeout_s
        inside_since = None
        while time.time() < deadline and not self.rospy.is_shutdown():
            if use_done and self.done_stamp > start_stamp:
                return True
            dist = self.distance_to(point)
            if dist <= tolerance:
                if inside_since is None:
                    inside_since = time.time()
                if time.time() - inside_since >= hold_s:
                    return True
            else:
                inside_since = None
            self.rospy.sleep(0.05)
        return False


def publish_takeoff_land(config, args, command):
    target = resolve_targets(config, args.uav)[0]
    configure_target_ros_env(config, target)

    import rospy
    from quadrotor_msgs.msg import TakeoffLand

    rospy.init_node(f"groundctrl_{target}_{command}", anonymous=True, disable_signals=True)
    topic_name = topic(config, "px4_takeoff_land", target)
    pub = rospy.Publisher(topic_name, TakeoffLand, queue_size=10)
    require_pub_connections(rospy, pub, topic_name, args.connect_timeout, f"{target} {command}")

    msg = TakeoffLand()
    msg.takeoff_land_cmd = TakeoffLand.TAKEOFF if command == "takeoff" else TakeoffLand.LAND
    for _ in range(args.repeat):
        pub.publish(msg)
        rospy.sleep(args.period)
    print(f"[groundctl] {target} {command} published to {topic_name}")


def make_pose_stamped(rospy, point):
    from geometry_msgs.msg import PoseStamped

    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "world"
    msg.pose.position.x = point[0]
    msg.pose.position.y = point[1]
    msg.pose.position.z = point[2]
    msg.pose.orientation.w = 1.0
    return msg


def publish_ego_goal(rospy, goal_pub, yaw_pub, point, yaw_deg):
    from std_msgs.msg import Float64

    stamp = rospy.Time.now()
    goal_msg = make_pose_stamped(rospy, point)
    goal_msg.header.stamp = stamp
    for _ in range(3):
        goal_pub.publish(goal_msg)
        if yaw_deg is not None:
            yaw_msg = Float64()
            yaw_msg.data = float(yaw_deg)
            yaw_pub.publish(yaw_msg)
        rospy.sleep(0.08)
    return stamp


def run_ego(config, args):
    target = resolve_targets(config, args.uav)[0]
    points = parse_points(args)
    configure_target_ros_env(config, target)

    import rospy
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Float64

    rospy.init_node(f"groundctrl_{target}_ego", anonymous=True, disable_signals=True)
    goal_topic = topic(config, "ego_goal", target)
    yaw_topic = topic(config, "ego_yaw", target)
    goal_pub = rospy.Publisher(goal_topic, PoseStamped, queue_size=10)
    yaw_pub = rospy.Publisher(yaw_topic, Float64, queue_size=10)
    observer = MissionObserver(rospy, config, target)
    require_pub_connections(rospy, goal_pub, goal_topic, args.connect_timeout, f"{target} ego")

    if args.ego_mode == "single":
        publish_ego_goal(rospy, goal_pub, yaw_pub, points[0], args.yaw_deg)
        print(f"[groundctl] {target} ego single -> {points[0]}")
        return

    if args.ego_mode == "timed":
        for idx, point in enumerate(points):
            publish_ego_goal(rospy, goal_pub, yaw_pub, point, args.yaw_deg)
            print(f"[groundctl] {target} ego timed {idx + 1}/{len(points)} -> {point}")
            if idx != len(points) - 1:
                rospy.sleep(args.interval)
        return

    if args.ego_mode == "reached":
        observer.wait_odom(args.odom_timeout)
        for idx, point in enumerate(points):
            stamp = publish_ego_goal(rospy, goal_pub, yaw_pub, point, args.yaw_deg)
            print(f"[groundctl] {target} ego reached {idx + 1}/{len(points)} -> {point}")
            ok = observer.wait_reached(
                point,
                stamp,
                args.tolerance,
                args.hold,
                args.timeout,
                not args.no_done,
            )
            if not ok:
                raise SystemExit(f"{target} did not reach {point} within {args.timeout:.1f}s")


def make_position_command(rospy, point, velocity, yaw_rad, trajectory_id):
    from quadrotor_msgs.msg import PositionCommand

    msg = PositionCommand()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "world"
    msg.position.x = point[0]
    msg.position.y = point[1]
    msg.position.z = point[2]
    msg.velocity.x = velocity[0]
    msg.velocity.y = velocity[1]
    msg.velocity.z = velocity[2]
    msg.yaw = yaw_rad
    msg.yaw_dot = 0.0
    msg.trajectory_id = trajectory_id
    msg.trajectory_flag = PositionCommand.TRAJECTORY_STATUS_READY
    return msg


def current_odom_point(observer):
    pos = observer.odom.pose.pose.position
    return (pos.x, pos.y, pos.z)


def publish_px4_for_duration(rospy, pub, observer, args, point, duration, trajectory_id):
    rate = rospy.Rate(args.rate)
    yaw_rad = yaw_rad_from_args(args)
    velocity = parse_vector(args.velocity, "velocity") if args.velocity else (0.0, 0.0, 0.0)
    deadline = time.time() + duration
    while time.time() < deadline and not rospy.is_shutdown():
        cmd_point = point
        if args.mode == "velocity":
            if observer.odom is None:
                observer.wait_odom(args.odom_timeout)
            cmd_point = current_odom_point(observer) if observer.odom is not None else (0.0, 0.0, 0.0)
        pub.publish(make_position_command(rospy, cmd_point, velocity, yaw_rad, trajectory_id))
        rate.sleep()


def publish_px4_until_reached(rospy, pub, observer, args, point, trajectory_id):
    if args.mode != "position":
        raise SystemExit("px4 reached mode only supports --mode position")
    if not observer.wait_odom(args.odom_timeout):
        raise SystemExit("No odometry received before px4 reached mission")

    rate = rospy.Rate(args.rate)
    yaw_rad = yaw_rad_from_args(args)
    velocity = parse_vector(args.velocity, "velocity") if args.velocity else (0.0, 0.0, 0.0)
    deadline = time.time() + args.timeout
    inside_since = None
    while time.time() < deadline and not rospy.is_shutdown():
        pub.publish(make_position_command(rospy, point, velocity, yaw_rad, trajectory_id))
        dist = observer.distance_to(point)
        if dist <= args.tolerance:
            if inside_since is None:
                inside_since = time.time()
            if time.time() - inside_since >= args.hold:
                return True
        else:
            inside_since = None
        rate.sleep()
    return False


def run_px4(config, args):
    target = resolve_targets(config, args.uav)[0]
    points_required = args.mode == "position"
    points = parse_points(args, required=points_required)
    if args.mode == "velocity" and not args.velocity:
        raise SystemExit("px4 --mode velocity requires --velocity vx,vy,vz")

    configure_target_ros_env(config, target)

    import rospy
    from quadrotor_msgs.msg import PositionCommand

    rospy.init_node(f"groundctrl_{target}_px4", anonymous=True, disable_signals=True)
    cmd_topic = topic(config, "px4_position_cmd", target)
    pub = rospy.Publisher(cmd_topic, PositionCommand, queue_size=20)
    observer = MissionObserver(rospy, config, target)
    require_pub_connections(rospy, pub, cmd_topic, args.connect_timeout, f"{target} px4 {args.px4_mode}")

    if args.px4_mode == "single":
        point = points[0] if points else (0.0, 0.0, 0.0)
        publish_px4_for_duration(rospy, pub, observer, args, point, args.duration, 1)
        print(f"[groundctl] {target} px4 single {args.mode} done")
        return

    if args.px4_mode == "timed":
        mission_points = points if points else [(0.0, 0.0, 0.0)]
        for idx, point in enumerate(mission_points):
            publish_px4_for_duration(rospy, pub, observer, args, point, args.interval, idx + 1)
            print(f"[groundctl] {target} px4 timed {idx + 1}/{len(mission_points)} done")
        return

    if args.px4_mode == "reached":
        for idx, point in enumerate(points):
            ok = publish_px4_until_reached(rospy, pub, observer, args, point, idx + 1)
            if not ok:
                raise SystemExit(f"{target} did not reach {point} within {args.timeout:.1f}s")
            publish_px4_for_duration(rospy, pub, observer, args, point, args.hold_after_reached, idx + 1)
            print(f"[groundctl] {target} px4 reached {idx + 1}/{len(points)} -> {point}")


def remote_ros_prefix(config, uav_name, use_namespace=False):
    uav_conf = config["uavs"][uav_name]
    project = shlex.quote(str(uav_conf["project"]))
    parts = [
        "source /opt/ros/noetic/setup.bash",
        f"cd {project}",
        "[ -f core/devel/setup.bash ] && source core/devel/setup.bash",
        f"export UAV_NAME={shlex.quote(uav_name)}",
        f"export ROS_MASTER_URI={shlex.quote(str(uav_conf['ros_master_uri']))}",
        f"export ROS_IP={shlex.quote(str(uav_conf['ip']))}",
        f"export ROS_HOSTNAME={shlex.quote(str(uav_conf['ip']))}",
    ]
    if use_namespace:
        parts.append(f"export ROS_NAMESPACE=/{shlex.quote(uav_name)}")
    else:
        parts.append("unset ROS_NAMESPACE")
    return " && ".join(parts)


def ssh_run(config, uav_name, command, check=False, timeout=None):
    ssh_host = config["uavs"][uav_name]["ssh_host"]
    ssh_cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=5",
        "-o", "ServerAliveInterval=5",
        "-o", "ServerAliveCountMax=2",
        ssh_host,
        "bash",
        "-lc",
        shlex.quote(command),
    ]
    try:
        result = subprocess.run(
            ssh_cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        output = output.rstrip()
        timeout_msg = f"[groundctl] ssh command timed out after {float(timeout):.1f}s"
        output = f"{output}\n{timeout_msg}" if output else timeout_msg
        result = subprocess.CompletedProcess(ssh_cmd, 124, output)
    if check and result.returncode != 0:
        raise SystemExit(result.stdout.strip() or f"ssh command failed: {uav_name}")
    return result


def remote_ssh_timeout(config, default=8.0):
    return float(config.get("runtime", {}).get("ssh_timeout_s", default))


def remote_ros_command(config, uav_name, ros_command, use_namespace=False):
    return f"{remote_ros_prefix(config, uav_name, use_namespace=use_namespace)} && {{ {ros_command}; }}"


def ros_name(ns, name):
    if ns:
        return f"/{ns}/{name}"
    return f"/{name}"


def start_health_spec(config, uav_name, component, use_namespace, no_ego=False):
    ns = uav_name if use_namespace else ""
    namespaced_topics = use_namespace
    if component == "roscore":
        return {
            "required_nodes": ["/rosout"],
            "required_topics": ["/rosout", "/rosout_agg"],
            "publisher_topics": [],
            "subscriber_topics": [],
            "sample_topics": [],
        }

    if component == "core":
        if namespaced_topics:
            camera_topic = f"/{uav_name}/camera/infra1/image_rect_raw"
            odom_topic = topic(config, "odom", uav_name)
            mavros_state_topic = topic(config, "mavros_state", uav_name)
            takeoff_land_topic = topic(config, "px4_takeoff_land", uav_name)
            position_cmd_topic = topic(config, "px4_position_cmd", uav_name)
        else:
            camera_topic = "/camera/infra1/image_rect_raw"
            odom_topic = "/vins_fusion/imu_propagate"
            mavros_state_topic = "/mavros/state"
            takeoff_land_topic = "/px4ctrl/takeoff_land"
            position_cmd_topic = "/position_cmd"

        required_nodes = [
            "/rosout",
            ros_name(ns, "camera/realsense2_camera_manager"),
            ros_name(ns, "mavros"),
            ros_name(ns, "vins_fusion"),
            ros_name(ns, "px4ctrl"),
        ]
        if not no_ego:
            required_nodes.extend([
                ros_name(ns, "drone_0_ego_planner_node"),
                ros_name(ns, "drone_0_traj_server"),
            ])

        return {
            "required_nodes": required_nodes,
            "required_topics": [
                camera_topic,
                odom_topic,
                mavros_state_topic,
                takeoff_land_topic,
                position_cmd_topic,
            ],
            "publisher_topics": [
                ("camera infra1 publisher", camera_topic),
                ("odom publisher", odom_topic),
                ("mavros state publisher", mavros_state_topic),
            ],
            "subscriber_topics": [
                ("takeoff/land subscriber", takeoff_land_topic),
                ("position command subscriber", position_cmd_topic),
            ],
            "sample_topics": [
                ("camera infra1 message", camera_topic),
                ("odom message", odom_topic),
                ("mavros state message", mavros_state_topic),
            ],
        }

    if component == "vins-sync":
        node_ns = f"/{uav_name}" if use_namespace else ""
        return {
            "required_nodes": [
                "/rosout",
                f"{node_ns}/{uav_name}_master_discovery",
                f"{node_ns}/{uav_name}_master_sync",
                f"{node_ns}/{uav_name}_vins_position_republisher",
            ],
            "required_topics": [topic(config, "vins_position", uav_name)],
            "publisher_topics": [
                ("vins position publisher", topic(config, "vins_position", uav_name)),
            ],
            "subscriber_topics": [],
            "sample_topics": [
                ("vins position message", topic(config, "vins_position", uav_name)),
            ],
        }

    raise SystemExit(f"Unknown health component: {component}")


def parse_remote_ros_index(output):
    nodes = set()
    topics = set()
    section = None
    for raw_line in (output or "").splitlines():
        line = raw_line.strip()
        if line == "__GROUNDCTL_NODES__":
            section = "nodes"
            continue
        if line == "__GROUNDCTL_TOPICS__":
            section = "topics"
            continue
        if line.startswith("__GROUNDCTL_RC__"):
            section = None
            continue
        if not line:
            continue
        if section == "nodes":
            nodes.add(line)
        elif section == "topics":
            topics.add(line)
    return nodes, topics


def collect_remote_ros_index(config, uav_name):
    ros_cmd = (
        "set +e; "
        "printf '__GROUNDCTL_NODES__\\n'; "
        "timeout 2 rosnode list 2>/dev/null; node_rc=$?; "
        "printf '__GROUNDCTL_TOPICS__\\n'; "
        "timeout 2 rostopic list 2>/dev/null; topic_rc=$?; "
        "printf '__GROUNDCTL_RC__ %s %s\\n' \"$node_rc\" \"$topic_rc\""
    )
    result = ssh_run(
        config,
        uav_name,
        remote_ros_command(config, uav_name, ros_cmd),
        timeout=remote_ssh_timeout(config, 8.0),
    )
    return parse_remote_ros_index(result.stdout), result


def topic_info_section_has_items(output, section_name):
    in_section = False
    section_headers = ("Publishers:", "Subscribers:")
    for raw_line in (output or "").splitlines():
        line = raw_line.strip()
        if line.startswith(section_name):
            in_section = True
            continue
        if any(line.startswith(header) for header in section_headers):
            in_section = False
            continue
        if in_section and line.startswith("* "):
            return True
    return False


def remote_topic_info(config, uav_name, topic_name, timeout_s):
    timeout_s = max(0.5, float(timeout_s))
    ros_cmd = f"timeout {timeout_s:.1f} rostopic info {shlex.quote(topic_name)} 2>/dev/null"
    return ssh_run(
        config,
        uav_name,
        remote_ros_command(config, uav_name, ros_cmd),
        timeout=timeout_s + remote_ssh_timeout(config, 5.0),
    )


def remote_topic_has_message(config, uav_name, topic_name, timeout_s):
    timeout_s = max(0.5, float(timeout_s))
    ros_cmd = f"timeout {timeout_s:.1f} rostopic echo -n 1 {shlex.quote(topic_name)} >/dev/null 2>&1"
    result = ssh_run(
        config,
        uav_name,
        remote_ros_command(config, uav_name, ros_cmd),
        timeout=timeout_s + remote_ssh_timeout(config, 5.0),
    )
    return result.returncode == 0


def print_check_line(ok, label, detail=""):
    print(format_check_line(ok, label, detail))


def format_check_line(ok, label, detail=""):
    status = "OK" if ok else "MISS"
    suffix = f" {detail}" if detail else ""
    return f"  [{status}] {label}{suffix}"


def build_start_feedback_report(config, uav_name, component, use_namespace, args):
    no_ego = component == "core" and getattr(args, "no_ego", False)
    spec = start_health_spec(config, uav_name, component, use_namespace, no_ego=no_ego)
    timeout_s = max(0.0, float(args.ready_timeout))
    poll_s = max(0.2, float(args.ready_poll))
    deadline = time.time() + timeout_s
    nodes = set()
    topics = set()
    ssh_ok = False

    lines = [f"[groundctl] {uav_name}: waiting up to {timeout_s:.1f}s for {component} feedback"]
    while True:
        (nodes, topics), result = collect_remote_ros_index(config, uav_name)
        ssh_ok = result.returncode == 0
        missing_nodes = [name for name in spec["required_nodes"] if name not in nodes]
        missing_topics = [name for name in spec["required_topics"] if name not in topics]
        roscore_ok = ssh_ok and ("/rosout" in nodes or "/rosout" in topics)
        basic_ok = roscore_ok and not missing_nodes and not missing_topics
        if basic_ok or time.time() >= deadline:
            break
        time.sleep(poll_s)

    missing_nodes = [name for name in spec["required_nodes"] if name not in nodes]
    missing_topics = [name for name in spec["required_topics"] if name not in topics]
    roscore_ok = ssh_ok and ("/rosout" in nodes or "/rosout" in topics)

    endpoint_ok = True
    endpoint_results = []
    for label, topic_name in spec["publisher_topics"]:
        info = remote_topic_info(config, uav_name, topic_name, args.sample_timeout)
        ok = info.returncode == 0 and topic_info_section_has_items(info.stdout, "Publishers:")
        endpoint_ok = endpoint_ok and ok
        endpoint_results.append((ok, label, topic_name))
    for label, topic_name in spec["subscriber_topics"]:
        info = remote_topic_info(config, uav_name, topic_name, args.sample_timeout)
        ok = info.returncode == 0 and topic_info_section_has_items(info.stdout, "Subscribers:")
        endpoint_ok = endpoint_ok and ok
        endpoint_results.append((ok, label, topic_name))

    sample_ok = True
    sample_results = []
    if not args.no_sample:
        for label, topic_name in spec["sample_topics"]:
            ok = remote_topic_has_message(config, uav_name, topic_name, args.sample_timeout)
            sample_ok = sample_ok and ok
            sample_results.append((ok, label, topic_name))

    basic_ok = roscore_ok and not missing_nodes and not missing_topics
    ready = basic_ok and endpoint_ok and sample_ok
    if ready:
        state = "READY"
    elif roscore_ok:
        state = "PARTIAL"
    else:
        state = "FAILED"

    lines.append(f"[groundctl] {uav_name} {component} feedback: {state}")
    lines.append(format_check_line(roscore_ok, "roscore reachable", config["uavs"][uav_name]["ros_master_uri"]))
    if missing_nodes:
        lines.append(format_check_line(False, "missing nodes", ", ".join(missing_nodes)))
    else:
        lines.append(format_check_line(True, "required nodes", f"{len(spec['required_nodes'])}/{len(spec['required_nodes'])}"))
    if missing_topics:
        lines.append(format_check_line(False, "missing topics", ", ".join(missing_topics)))
    else:
        lines.append(format_check_line(True, "required topics", f"{len(spec['required_topics'])}/{len(spec['required_topics'])}"))
    for ok, label, topic_name in endpoint_results:
        lines.append(format_check_line(ok, label, topic_name))
    for ok, label, topic_name in sample_results:
        lines.append(format_check_line(ok, label, f"{topic_name} within {float(args.sample_timeout):.1f}s"))
    return ready, "\n".join(lines)


def wait_for_start_feedback(config, uav_name, component, use_namespace, args):
    ready, report = build_start_feedback_report(config, uav_name, component, use_namespace, args)
    print(report)
    return ready


def run_status(config, args):
    for target in resolve_targets(config, args.uav):
        cmd = (
            f"{remote_ros_prefix(config, target)} && "
            "printf 'host=' && hostname && "
            "printf 'ROS_MASTER_URI=' && printf '%s\\n' \"$ROS_MASTER_URI\" && "
            "timeout 2 rostopic list 2>/dev/null | sed -n '1,80p' || true"
        )
        result = ssh_run(config, target, cmd)
        print(f"--- {target} ({config['uavs'][target]['ip']}) ---")
        print(result.stdout.rstrip() if result.stdout else f"ssh exit={result.returncode}")


def run_remote(config, args):
    command = " ".join(shlex.quote(item) for item in args.remote_command)
    for target in resolve_targets(config, args.uav):
        if args.ros_env:
            command_to_run = f"{remote_ros_prefix(config, target)} && {command}"
        else:
            command_to_run = command
        result = ssh_run(config, target, command_to_run)
        print(f"--- {target} exit={result.returncode} ---")
        if result.stdout:
            print(result.stdout.rstrip())


def run_topic(config, args):
    for target in resolve_targets(config, args.uav):
        if args.topic_command == "list":
            ros_cmd = "rostopic list"
        elif args.topic_command == "info":
            ros_cmd = f"rostopic info {shlex.quote(args.topic)}"
        elif args.topic_command == "echo":
            ros_cmd = f"rostopic echo -n {int(args.count)} {shlex.quote(args.topic)}"
        elif args.topic_command == "pub":
            rate_arg = f"-r {float(args.rate)}" if args.continuous else "-1"
            ros_cmd = (
                f"rostopic pub {rate_arg} {shlex.quote(args.topic)} "
                f"{shlex.quote(args.msg_type)} {shlex.quote(args.yaml_msg)}"
            )
        else:
            raise SystemExit(f"Unknown topic command: {args.topic_command}")
        result = ssh_run(config, target, f"{remote_ros_prefix(config, target)} && {ros_cmd}")
        print(f"--- {target} exit={result.returncode} ---")
        if result.stdout:
            print(result.stdout.rstrip())


def format_start_result(target, component, result):
    lines = [f"--- {target} {component} exit={result.returncode} ---"]
    output = (result.stdout or "").strip()
    if not output:
        return "\n".join(lines)
    output_lines = [line.strip() for line in output.splitlines() if line.strip()]
    pid_lines = [line for line in output_lines if line.isdigit()]
    if pid_lines:
        lines.append(f"pid={pid_lines[-1]}")
        extra_lines = [line for line in output_lines if line not in pid_lines]
        if extra_lines:
            lines.extend(extra_lines)
    else:
        lines.append(output)
    return "\n".join(lines)


def print_start_result(target, component, result):
    print(format_start_result(target, component, result))


def start_launch_command(config, target, start_target, use_ns, no_ego=False):
    log_dir = config.get("runtime", {}).get("remote_log_dir", "logs/groundctrl")
    prefix = remote_ros_prefix(config, target, use_namespace=use_ns)
    mkdir = f"mkdir -p {shlex.quote(log_dir)}"
    sudo_ttyacm = "printf ' \\n' | sudo -S -p '' chmod 777 /dev/ttyACM0 >/dev/null 2>&1 || true"

    if start_target == "roscore":
        launch = (
            f"nohup roscore > {shlex.quote(log_dir)}/roscore.log 2>&1 < /dev/null & "
            "pid=$!; disown \"$pid\" 2>/dev/null || true; printf '%s\\n' \"$pid\""
        )
    elif start_target == "core":
        if no_ego:
            no_ego_command = (
                "if grep -q START_EGO_PLANNER ./core/shfiles/basectrl.sh; then "
                "START_EGO_PLANNER=0 ./core/shfiles/basectrl.sh; "
                "else "
                "sed '/roslaunch ego_planner single_run_in_exp.launch/d' ./core/shfiles/basectrl.sh | bash; "
                "fi"
            )
            launch_script = f"bash -lc {shlex.quote(no_ego_command)}"
        else:
            launch_script = "./core/shfiles/basectrl.sh"
        launch = (
            f"{sudo_ttyacm}; "
            f"nohup {launch_script} > {shlex.quote(log_dir)}/core.log 2>&1 < /dev/null & "
            "pid=$!; disown \"$pid\" 2>/dev/null || true; printf '%s\\n' \"$pid\""
        )
    elif start_target == "vins-sync":
        uav_id = config["uavs"][target]["id"]
        launch = (
            "nohup roslaunch swarm_position_bridge uav_vins_sync.launch "
            f"uav_id:={uav_id} uav_ns:={shlex.quote(target)} "
            f"> {shlex.quote(log_dir)}/vins_sync.log 2>&1 < /dev/null & "
            "pid=$!; disown \"$pid\" 2>/dev/null || true; printf '%s\\n' \"$pid\""
        )
    else:
        raise SystemExit(f"Unknown start target: {start_target}")
    return f"{prefix} && {mkdir} && {{ {launch}; }}"


def run_start_launch_job(config, target, args, use_ns):
    result = ssh_run(
        config,
        target,
        start_launch_command(
            config,
            target,
            args.start_target,
            use_ns,
            no_ego=getattr(args, "no_ego", False),
        ),
        timeout=float(args.launch_timeout),
    )
    return {
        "target": target,
        "use_ns": use_ns,
        "result": result,
        "output": format_start_result(target, args.start_target, result),
    }


def run_start_ready_job(config, target, args, use_ns):
    ready, output = build_start_feedback_report(
        config,
        target,
        args.start_target,
        use_ns,
        args,
    )
    return {
        "target": target,
        "ready": ready,
        "output": output.rstrip(),
    }


def run_jobs_in_thread_pool(targets, max_workers, job_func):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_by_target = {
            executor.submit(job_func, target): target
            for target in targets
        }
        for future in as_completed(future_by_target):
            target = future_by_target[future]
            try:
                results[target] = future.result()
            except Exception as exc:
                results[target] = {
                    "target": target,
                    "error": exc,
                    "output": f"--- {target} error ---\n{exc}",
                }
    return results


def start_use_namespace(config, args):
    if args.namespace is not None:
        return args.namespace
    return bool(config.get("runtime", {}).get("set_ros_namespace_on_remote_start", True))


def run_start(config, args):
    launch_failed = 0
    not_ready = []
    launched = []
    targets = resolve_targets(config, args.uav)
    use_ns_by_target = {
        target: start_use_namespace(config, args)
        for target in targets
    }
    max_workers = max(1, min(int(args.jobs), len(targets)))

    print(f"[groundctl] launching {args.start_target} on {len(targets)} target(s) with {max_workers} worker(s)")
    sys.stdout.flush()
    launch_results = run_jobs_in_thread_pool(
        targets,
        max_workers,
        lambda target: run_start_launch_job(config, target, args, use_ns_by_target[target]),
    )
    for target in targets:
        job = launch_results[target]
        print(job["output"])
        sys.stdout.flush()
        if "error" in job:
            launch_failed = 1 if launch_failed == 0 else launch_failed
            continue
        result = job["result"]
        if result.returncode != 0:
            launch_failed = result.returncode if launch_failed == 0 else launch_failed
            continue
        launched.append((target, job["use_ns"]))

    if not args.no_wait_ready:
        ready_targets = [target for target, _use_ns in launched]
        ready_use_ns_by_target = dict(launched)
        print(f"[groundctl] checking {args.start_target} readiness on {len(ready_targets)} target(s) with {max_workers} worker(s)")
        sys.stdout.flush()
        ready_results = run_jobs_in_thread_pool(
            ready_targets,
            max_workers,
            lambda target: run_start_ready_job(config, target, args, ready_use_ns_by_target[target]),
        )
        for target in ready_targets:
            job = ready_results[target]
            if job["output"]:
                print(job["output"])
            sys.stdout.flush()
            if job.get("error") is not None or not job.get("ready", False):
                not_ready.append(target)

    if launch_failed:
        raise SystemExit(launch_failed)
    if args.strict_ready and not_ready:
        raise SystemExit(2)


def run_check(config, args):
    not_ready = []
    use_ns = start_use_namespace(config, args)
    for target in resolve_targets(config, args.uav):
        ready = wait_for_start_feedback(config, target, args.check_target, use_ns, args)
        if not ready:
            not_ready.append(target)
    if args.strict_ready and not_ready:
        raise SystemExit(2)


def stop_process_pattern(stop_target):
    base = [
        "[r]oslaunch",
        "[r]osout",
        "[n]odelet",
    ]
    groups = {
        "roscore": [
            "[r]oscore",
            "[r]osmaster",
            "[r]osout",
        ],
        "vins-sync": [
            "[m]aster_discovery",
            "[m]aster_sync",
            "[v]ins_position_republisher.py",
        ],
        "core": base + [
            "[b]asectrl.sh",
            "[r]oscore",
            "[r]osmaster",
            "[m]avros_node",
            "[v]ins_node",
            "[p]x4ctrl_node",
            "[e]go_planner_node",
            "[t]raj_server",
            "[r]ealsense2_camera",
            "[m]aster_discovery",
            "[m]aster_sync",
            "[v]ins_position_republisher.py",
        ],
        "all": base + [
            "[b]asectrl.sh",
            "[r]oscore",
            "[r]osmaster",
            "[r]osrun",
            "[r]ostopic",
            "[r]osbag",
            "[r]qt",
            "[r]viz",
            "[m]avros_node",
            "[v]ins_node",
            "[p]x4ctrl_node",
            "[e]go_planner_node",
            "[t]raj_server",
            "[r]ealsense2_camera",
            "[m]aster_discovery",
            "[m]aster_sync",
            "[v]ins_position_republisher.py",
        ],
    }
    return "|".join(groups[stop_target])


def stop_remote_command(args):
    pattern = stop_process_pattern(args.stop_target)
    grace = max(0.0, float(args.grace))
    commands = [
        "set +e",
        f"PAT={shlex.quote(pattern)}",
        "printf '[groundctl] stopping %s on %s\\n' " +
        f"{shlex.quote(args.stop_target)} \"$UAV_NAME\"",
    ]
    if not args.no_rosnode_kill and args.stop_target in ("core", "all", "vins-sync"):
        commands.append("timeout 5 rosnode kill -a >/dev/null 2>&1 || true")
    commands.extend([
        f"pkill -INT -u \"$(id -u)\" -f \"$PAT\" >/dev/null 2>&1 || true",
        f"sleep {grace:.2f}",
        f"pkill -TERM -u \"$(id -u)\" -f \"$PAT\" >/dev/null 2>&1 || true",
        "sleep 1",
    ])
    if args.force:
        commands.append("pkill -KILL -u \"$(id -u)\" -f \"$PAT\" >/dev/null 2>&1 || true")
        commands.append("sleep 0.5")
    commands.extend([
        "remaining=$(pgrep -u \"$(id -u)\" -af \"$PAT\" || true)",
        "if [ -n \"$remaining\" ]; then "
        "printf '[groundctl] remaining processes:\\n%s\\n' \"$remaining\"; exit 2; "
        "else printf '[groundctl] stopped\\n'; exit 0; fi",
    ])
    return "; ".join(commands)


def format_stop_result(target, stop_target, result):
    lines = [f"--- {target} stop {stop_target} exit={result.returncode} ---"]
    output = (result.stdout or "").rstrip()
    if output:
        lines.append(output)
    return "\n".join(lines)


def run_stop_job(config, target, args):
    result = ssh_run(
        config,
        target,
        remote_ros_command(config, target, stop_remote_command(args)),
        timeout=float(args.stop_timeout),
    )
    return {
        "target": target,
        "result": result,
        "output": format_stop_result(target, args.stop_target, result),
    }


def run_stop(config, args):
    failed = 0
    targets = resolve_targets(config, args.uav)
    max_workers = max(1, min(int(args.jobs), len(targets)))

    print(f"[groundctl] stopping {args.stop_target} on {len(targets)} target(s) with {max_workers} worker(s)")
    sys.stdout.flush()
    stop_results = run_jobs_in_thread_pool(
        targets,
        max_workers,
        lambda target: run_stop_job(config, target, args),
    )
    for target in targets:
        job = stop_results[target]
        print(job["output"])
        sys.stdout.flush()
        if "error" in job:
            failed = 1 if failed == 0 else failed
            continue
        result = job["result"]
        if result.returncode != 0:
            failed = result.returncode if failed == 0 else failed
    if failed and not args.ignore_remaining:
        raise SystemExit(failed)


def add_target_arg(parser):
    parser.add_argument("--uav", default="all", help="Target UAV: uav0, uav1, uav2, or comma list/all")


def add_common_ros_args(parser):
    parser.add_argument("--connect-timeout", type=float, default=3.0)


def add_namespace_args(parser):
    ns_group = parser.add_mutually_exclusive_group()
    ns_group.add_argument("--namespace", dest="namespace", action="store_true")
    ns_group.add_argument("--no-namespace", dest="namespace", action="store_false")
    parser.set_defaults(namespace=None)


def add_readiness_args(parser):
    parser.add_argument("--ready-timeout", type=float, default=30.0)
    parser.add_argument("--ready-poll", type=float, default=2.0)
    parser.add_argument("--sample-timeout", type=float, default=2.0)
    parser.add_argument("--no-sample", action="store_true", help="Skip rostopic echo message samples")
    parser.add_argument("--strict-ready", action="store_true", help="Exit non-zero if feedback is not READY")


def add_point_args(parser):
    parser.add_argument("--points", dest="points_text", default="", help="Points: 'x,y,z;x,y,z'")
    parser.add_argument("point_args", nargs="*", help="Points such as 1,0,0.6 2,0,0.6")


def build_parser():
    parser = argparse.ArgumentParser(description="MUAV ground station control tool")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="groundctrl.yaml path")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Check SSH and remote ROS topic visibility")
    add_target_arg(status)
    status.set_defaults(func=run_status)

    remote = sub.add_parser("remote", help="Run a shell command on target UAVs")
    add_target_arg(remote)
    remote.add_argument("--ros-env", action="store_true", help="Source ROS and MUAV setup first")
    remote.add_argument("remote_command", nargs=argparse.REMAINDER)
    remote.set_defaults(func=run_remote)

    stop = sub.add_parser("stop", help="Stop remote ROS components")
    add_target_arg(stop)
    stop.add_argument("stop_target", choices=("core", "roscore", "vins-sync", "all"))
    stop.add_argument("--force", action="store_true", help="Send SIGKILL if graceful stop leaves processes")
    stop.add_argument("--grace", type=float, default=2.0, help="Seconds to wait between INT and TERM")
    stop.add_argument("--jobs", type=int, default=8, help="Maximum concurrent UAV stop jobs")
    stop.add_argument("--stop-timeout", type=float, default=20.0, help="Seconds to wait for each remote SSH stop command")
    stop.add_argument("--no-rosnode-kill", action="store_true", help="Skip rosnode kill -a before process cleanup")
    stop.add_argument("--ignore-remaining", action="store_true", help="Return success even if matching processes remain")
    stop.set_defaults(func=run_stop)

    topic_parser = sub.add_parser("topic", help="Run rostopic on target UAVs")
    add_target_arg(topic_parser)
    topic_sub = topic_parser.add_subparsers(dest="topic_command", required=True)
    topic_sub.add_parser("list").set_defaults(func=run_topic)
    info = topic_sub.add_parser("info")
    info.add_argument("topic")
    info.set_defaults(func=run_topic)
    echo = topic_sub.add_parser("echo")
    echo.add_argument("topic")
    echo.add_argument("-n", "--count", type=int, default=1)
    echo.set_defaults(func=run_topic)
    pub = topic_sub.add_parser("pub")
    pub.add_argument("topic")
    pub.add_argument("msg_type")
    pub.add_argument("yaml_msg")
    pub.add_argument("--continuous", action="store_true")
    pub.add_argument("--rate", type=float, default=1.0)
    pub.set_defaults(func=run_topic)

    start = sub.add_parser("start", help="Start remote ROS components")
    add_target_arg(start)
    start.add_argument("start_target", choices=("roscore", "core", "vins-sync"))
    add_namespace_args(start)
    add_readiness_args(start)
    start.add_argument("--jobs", type=int, default=8, help="Maximum concurrent UAV start/check jobs")
    start.add_argument("--launch-timeout", type=float, default=10.0, help="Seconds to wait for each remote SSH launch command")
    start.add_argument("--no-wait-ready", action="store_true", help="Only launch remotely; do not check readiness")
    start.add_argument("--no-ego", action="store_true", help="Start core without ego_planner")
    start.set_defaults(func=run_start)

    check = sub.add_parser("check", help="Check remote ROS component readiness without starting")
    add_target_arg(check)
    check.add_argument("check_target", choices=("roscore", "core", "vins-sync"))
    add_namespace_args(check)
    add_readiness_args(check)
    check.add_argument("--no-ego", action="store_true", help="Check core readiness without requiring ego_planner")
    check.set_defaults(func=run_check)

    takeoff = sub.add_parser("takeoff", help="Publish px4ctrl takeoff command")
    add_target_arg(takeoff)
    add_common_ros_args(takeoff)
    takeoff.add_argument("--repeat", type=int, default=3)
    takeoff.add_argument("--period", type=float, default=0.2)
    takeoff.set_defaults(func=lambda config, args: publish_takeoff_land(config, args, "takeoff"))

    land = sub.add_parser("land", help="Publish px4ctrl land command")
    add_target_arg(land)
    add_common_ros_args(land)
    land.add_argument("--repeat", type=int, default=3)
    land.add_argument("--period", type=float, default=0.2)
    land.set_defaults(func=lambda config, args: publish_takeoff_land(config, args, "land"))

    ego = sub.add_parser("ego", help="EGO planner missions")
    ego_sub = ego.add_subparsers(dest="ego_mode", required=True)
    for name in ("single", "timed", "reached"):
        mode = ego_sub.add_parser(name)
        add_target_arg(mode)
        add_common_ros_args(mode)
        mode.add_argument("--yaw-deg", type=float, default=None)
        add_point_args(mode)
        mode.set_defaults(func=run_ego)
        if name == "timed":
            mode.add_argument("--interval", type=float, default=5.0)
        if name == "reached":
            mode.add_argument("--timeout", type=float, default=60.0)
            mode.add_argument("--tolerance", type=float, default=0.30)
            mode.add_argument("--hold", type=float, default=0.30)
            mode.add_argument("--odom-timeout", type=float, default=5.0)
            mode.add_argument("--no-done", action="store_true")

    px4 = sub.add_parser("px4", help="Direct px4ctrl PositionCommand missions")
    px4_sub = px4.add_subparsers(dest="px4_mode", required=True)
    for name in ("single", "timed", "reached"):
        mode = px4_sub.add_parser(name)
        add_target_arg(mode)
        add_common_ros_args(mode)
        mode.add_argument("--mode", choices=("position", "velocity"), default="position")
        mode.add_argument("--velocity", default="", help="Velocity feed-forward or velocity command: vx,vy,vz")
        mode.add_argument("--yaw-deg", type=float, default=None)
        mode.add_argument("--rate", type=float, default=20.0)
        mode.add_argument("--odom-timeout", type=float, default=5.0)
        add_point_args(mode)
        mode.set_defaults(func=run_px4)
        if name == "single":
            mode.add_argument("--duration", type=float, default=5.0)
        if name == "timed":
            mode.add_argument("--interval", type=float, default=5.0)
        if name == "reached":
            mode.add_argument("--timeout", type=float, default=60.0)
            mode.add_argument("--tolerance", type=float, default=0.30)
            mode.add_argument("--hold", type=float, default=0.30)
            mode.add_argument("--hold-after-reached", type=float, default=1.0)

    return parser


def main():
    parser = build_parser()
    args = parse_args_allowing_negative_vectors(parser)
    config = load_config(args.config)

    if args.command == "remote" and not args.remote_command:
        raise SystemExit("remote requires a command")

    fan_out_if_needed(config, args)
    args.func(config, args)


if __name__ == "__main__":
    main()
