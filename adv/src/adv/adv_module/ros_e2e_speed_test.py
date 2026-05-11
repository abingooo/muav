#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import statistics
import sys
import tempfile
import threading
from pathlib import Path
from typing import List

import yaml

from .ros_adapter import AdvRosAdapter, DEFAULT_MODEL_CONFIG_PATH, _load_yaml, _section
from .ros_e2e_smoke_test import (
    _default_states,
    _input_entries,
    _make_odom_msg,
    _make_point_msg,
)


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Measure ADV ROS adapter end-to-end throughput limit.")
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG_PATH, help="Path to model_config.yaml.")
    parser.add_argument("--duration", type=float, default=10.0, help="Measurement duration in seconds.")
    parser.add_argument("--warmup", type=float, default=1.0, help="Warmup seconds before collecting samples.")
    parser.add_argument("--input-rate", type=float, default=200.0, help="Fake input publish rate in Hz.")
    parser.add_argument(
        "--adapter-rate",
        type=float,
        default=1000.0,
        help="Temporary adapter publish_rate_hz used for limit testing.",
    )
    parser.add_argument("--startup-wait", type=float, default=2.0, help="Seconds to wait after starting the adapter.")
    parser.add_argument(
        "--message-type",
        choices=("point", "odom"),
        default="point",
        help="Fake input message type: point=geometry_msgs/PointStamped, odom=nav_msgs/Odometry.",
    )
    parser.add_argument(
        "--no-start-adapter",
        action="store_true",
        help="Do not start an in-process adapter; measure an already-running adapter.",
    )
    parser.add_argument(
        "--min-rate",
        type=float,
        default=0.0,
        help="Optional minimum measured rate. The test fails if measured rate is below this value.",
    )
    return parser.parse_args(argv)


def _interval_stats(stamps: List[float]):
    intervals = [b - a for a, b in zip(stamps[:-1], stamps[1:])]
    if not intervals:
        return None
    return {
        "min": min(intervals),
        "mean": statistics.mean(intervals),
        "max": max(intervals),
        "stddev": statistics.pstdev(intervals) if len(intervals) > 1 else 0.0,
    }


def _resolve_relative_path(path: str, base_dir: Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(base_dir / candidate)


def _write_limit_config(config: dict, source_model_config_path: str, adapter_rate: float) -> str:
    limit_config = copy.deepcopy(config)
    adapter = _section(limit_config, "adapter")
    adapter["publish_rate_hz"] = float(adapter_rate)
    game_end = limit_config.get("game_end", {})
    if isinstance(game_end, dict):
        game_end["enabled"] = False
        limit_config["game_end"] = game_end
    source_dir = Path(source_model_config_path).resolve().parent
    adapter["inference_config_path"] = _resolve_relative_path(
        str(adapter.get("inference_config_path", "inference_config.yaml")),
        source_dir,
    )
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix="_adv_model_config.yaml",
        delete=False,
    )
    with tmp:
        yaml.safe_dump(limit_config, tmp, allow_unicode=True, sort_keys=False)
    return tmp.name


def _start_adapter_thread(config_path: str):
    adapter = AdvRosAdapter(model_config_path=config_path)
    thread = threading.Thread(target=adapter.spin, name="adv_adapter_speed_limit", daemon=True)
    thread.start()
    return thread


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
    configured_rate = float(adapter.get("publish_rate_hz", 20.0))

    rospy.init_node("adv_ros_e2e_speed_limit_test", anonymous=True)

    if args.no_start_adapter:
        rospy.loginfo("measuring external adapter; configured model_config publish_rate_hz=%.2f Hz", configured_rate)
    else:
        limit_config_path = _write_limit_config(config, args.model_config, float(args.adapter_rate))
        rospy.loginfo("starting in-process adapter with publish_rate_hz=%.2f Hz", args.adapter_rate)
        rospy.loginfo("temporary model_config: %s", limit_config_path)
        _start_adapter_thread(limit_config_path)
        rospy.sleep(float(args.startup_wait))

    msg_cls = PointStamped if args.message_type == "point" else Odometry
    publishers = {
        role: rospy.Publisher(topic, msg_cls, queue_size=10)
        for role, topic in input_topics.items()
    }

    output_stamps: List[float] = []
    collecting = {"enabled": False}

    def output_callback(msg: PositionCommand):
        if collecting["enabled"]:
            output_stamps.append(float(rospy.Time.now().to_sec()))

    rospy.Subscriber(output_topic, PositionCommand, output_callback, queue_size=100)

    rospy.loginfo("speed limit test publishing %s inputs at %.2f Hz", args.message_type, args.input_rate)
    rospy.loginfo("measuring output topic: %s", rospy.resolve_name(output_topic))
    rospy.loginfo("measurement duration: %.2fs after %.2fs warmup", args.duration, args.warmup)

    make_msg = _make_point_msg if args.message_type == "point" else _make_odom_msg
    msg_type = PointStamped if args.message_type == "point" else Odometry
    rate = rospy.Rate(float(args.input_rate))

    start = float(rospy.Time.now().to_sec())
    collect_start = start + float(args.warmup)
    deadline = collect_start + float(args.duration)

    while not rospy.is_shutdown() and float(rospy.Time.now().to_sec()) < deadline:
        now = float(rospy.Time.now().to_sec())
        if now >= collect_start:
            collecting["enabled"] = True

        stamp = rospy.Time.now()
        for role, publisher in publishers.items():
            publisher.publish(make_msg(msg_type, role, states[role], stamp))
        rate.sleep()

    sample_count = len(output_stamps)
    measured_span = max(1e-9, output_stamps[-1] - output_stamps[0]) if sample_count >= 2 else 0.0
    measured_rate = (sample_count - 1) / measured_span if sample_count >= 2 else 0.0
    stats = _interval_stats(output_stamps)

    rospy.loginfo("samples: %d", sample_count)
    rospy.loginfo("measured output rate limit: %.3f Hz", measured_rate)
    if stats is not None:
        rospy.loginfo(
            "interval sec: min=%.4f mean=%.4f max=%.4f stddev=%.4f",
            stats["min"],
            stats["mean"],
            stats["max"],
            stats["stddev"],
        )
        rospy.loginfo("fastest observed interval rate: %.3f Hz", 1.0 / stats["min"] if stats["min"] > 0 else 0.0)
    else:
        rospy.logerr("FAIL: fewer than 2 output samples received")
        return 2

    if float(args.min_rate) > 0.0 and measured_rate < float(args.min_rate):
        rospy.logerr("FAIL: measured rate %.3f Hz < minimum %.3f Hz", measured_rate, float(args.min_rate))
        return 1

    rospy.loginfo("PASS: measured end-to-end output limit %.3f Hz", measured_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
