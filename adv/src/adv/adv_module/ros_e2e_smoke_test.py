#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .ros_adapter import (
    ALL_ROLES,
    DEFAULT_MODEL_CONFIG_PATH,
    _as_float3,
    _load_yaml,
    _section,
)


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Publish fake VINS inputs and wait for an ADV position command.")
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG_PATH, help="Path to model_config.yaml.")
    parser.add_argument("--timeout", type=float, default=8.0, help="Seconds to wait for a valid output command.")
    parser.add_argument("--rate", type=float, default=20.0, help="Input publish rate in Hz.")
    parser.add_argument(
        "--message-type",
        choices=("point", "odom"),
        default="point",
        help="Fake input message type: point=geometry_msgs/PointStamped, odom=nav_msgs/Odometry.",
    )
    return parser.parse_args(argv)


def _input_entries(config: Dict[str, Any]):
    entries = config.get("inputs", [])
    if not isinstance(entries, list):
        raise ValueError("model_config.yaml: inputs 必须是 list")
    by_role = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("model_config.yaml: inputs 中每项必须是 mapping")
        role = str(entry.get("role", ""))
        topic = str(entry.get("topic", ""))
        if role not in ALL_ROLES:
            raise ValueError(f"未知 role: {role}")
        if not topic:
            raise ValueError(f"{role} 缺少 topic")
        by_role[role] = topic
    missing = [role for role in ALL_ROLES if role not in by_role]
    if missing:
        raise ValueError(f"inputs 缺少角色: {missing}")
    return by_role


def _default_states(config: Dict[str, Any]):
    defaults = _section(config, "defaults")
    states = _section(defaults, "states")
    out = {}
    for role in ALL_ROLES:
        state = states.get(role)
        if not isinstance(state, dict):
            raise ValueError(f"defaults.states.{role} 缺失")
        out[role] = {
            "position": _as_float3(state.get("position", [0.0, 0.0, None]), name=f"{role}.position"),
            "velocity": _as_float3(state.get("velocity", [0.0, 0.0, 0.0]), name=f"{role}.velocity"),
        }
    return out


def _make_point_msg(PointStamped, role: str, state: Dict[str, Any], stamp):
    msg = PointStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = "world"
    msg.point.x = float(state["position"][0])
    msg.point.y = float(state["position"][1])
    msg.point.z = float(state["position"][2])
    return msg


def _make_odom_msg(Odometry, role: str, state: Dict[str, Any], stamp):
    msg = Odometry()
    msg.header.stamp = stamp
    msg.header.frame_id = "world"
    msg.child_frame_id = role
    msg.pose.pose.position.x = float(state["position"][0])
    msg.pose.pose.position.y = float(state["position"][1])
    msg.pose.pose.position.z = float(state["position"][2])
    msg.pose.pose.orientation.w = 1.0
    msg.twist.twist.linear.x = float(state["velocity"][0])
    msg.twist.twist.linear.y = float(state["velocity"][1])
    msg.twist.twist.linear.z = float(state["velocity"][2])
    return msg


def _is_finite_vector(values):
    return all(math.isfinite(float(value)) for value in values)


def main() -> int:
    import rospy
    from geometry_msgs.msg import PointStamped
    from nav_msgs.msg import Odometry
    from quadrotor_msgs.msg import PositionCommand

    args = _parse_args(rospy.myargv(argv=sys.argv)[1:])
    config = _load_yaml(args.model_config)
    adapter = _section(config, "adapter")
    input_topics = _input_entries(config)
    states = _default_states(config)

    output_topic = str(adapter.get("output_topic", "/adv/position_cmd"))
    expected_z = float(adapter.get("output_default_height", 1.0))
    expected_vz = float(adapter.get("output_velocity_z", 0.0))

    rospy.init_node("adv_ros_e2e_smoke_test", anonymous=True)

    msg_cls = PointStamped if args.message_type == "point" else Odometry
    publishers = {
        role: rospy.Publisher(topic, msg_cls, queue_size=10)
        for role, topic in input_topics.items()
    }

    received: Dict[str, Optional[PositionCommand]] = {"msg": None}

    def output_callback(msg):
        received["msg"] = msg

    rospy.Subscriber(output_topic, PositionCommand, output_callback, queue_size=10)

    rospy.loginfo("smoke test publishing %s inputs", args.message_type)
    for role, topic in input_topics.items():
        rospy.loginfo("input: %s -> %s", role, rospy.resolve_name(topic))
    rospy.loginfo("waiting output: %s", rospy.resolve_name(output_topic))

    deadline = rospy.Time.now().to_sec() + float(args.timeout)
    rate = rospy.Rate(float(args.rate))
    make_msg = _make_point_msg if args.message_type == "point" else _make_odom_msg
    msg_type = PointStamped if args.message_type == "point" else Odometry

    while not rospy.is_shutdown() and rospy.Time.now().to_sec() < deadline:
        stamp = rospy.Time.now()
        for role, publisher in publishers.items():
            publisher.publish(make_msg(msg_type, role, states[role], stamp))

        msg = received["msg"]
        if msg is not None:
            velocity = [msg.velocity.x, msg.velocity.y, msg.velocity.z]
            if not _is_finite_vector([msg.position.x, msg.position.y, msg.position.z, *velocity]):
                rospy.logerr("FAIL: output contains non-finite values")
                return 2
            if abs(float(msg.position.z) - expected_z) > 1e-6:
                rospy.logerr("FAIL: position.z %.6f != expected %.6f", msg.position.z, expected_z)
                return 3
            if abs(float(msg.velocity.z) - expected_vz) > 1e-6:
                rospy.logerr("FAIL: velocity.z %.6f != expected %.6f", msg.velocity.z, expected_vz)
                return 4
            rospy.loginfo(
                "PASS: got %s position=(%.3f, %.3f, %.3f) velocity=(%.3f, %.3f, %.3f)",
                rospy.resolve_name(output_topic),
                msg.position.x,
                msg.position.y,
                msg.position.z,
                msg.velocity.x,
                msg.velocity.y,
                msg.velocity.z,
            )
            return 0

        rate.sleep()

    rospy.logerr("FAIL: no valid output from %s within %.1fs", rospy.resolve_name(output_topic), args.timeout)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
