#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from .coordinate_transform import CoordinateTransform
from .game_end import GameEndConfig, GameEndMonitor, GameEndStatus
from .inference_framework import (
    InferenceEngine,
    InferenceSnapshot,
    VehicleState,
)


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_CONFIG_PATH = str(MODULE_DIR / "model_config.yaml")
DEFENDER_ROLES = ("defender_0", "defender_1", "defender_2", "defender_3")
ALL_ROLES = DEFENDER_ROLES + ("enemy",)
TOP_LEVEL_KEYS = {"adapter", "defaults", "coordinate_transforms", "inputs", "position_command", "game_end"}
ADAPTER_KEYS = {
    "output_topic",
    "plan_point_enabled",
    "plan_point_topic",
    "plan_point_frame_id",
    "output_role",
    "output_default_height",
    "output_velocity_z",
    "output_frame",
    "publish_rate_hz",
    "stale_timeout_sec",
    "command_position_dt",
    "output_frame_id",
    "estimate_velocity_from_position",
    "inference_config_path",
    "active_defender_roles",
}
STATE_KEYS = {"position", "velocity"}
TRANSFORM_KEYS = {"translation", "yaw_deg"}
INPUT_KEYS = {"role", "topic", "message_type"}
POSITION_COMMAND_KEYS = {"kx", "kv", "yaw", "yaw_dot", "trajectory_id"}
GAME_END_KEYS = {
    "enabled",
    "capture_distance_m",
    "asset_distance_m",
    "hold_duration_sec",
    "asset_origin",
    "out_of_bounds_enabled",
    "world_bounds",
    "command_topics",
}
WORLD_BOUNDS_KEYS = {"x", "y", "z"}


@dataclass
class TopicState:
    position: np.ndarray
    velocity: np.ndarray
    stamp_sec: Optional[float] = None
    received: bool = False

def _load_yaml(path: str) -> Dict[str, Any]:
    return _load_yaml_mapping(path, description="model_config")


def _load_yaml_mapping(path: str, *, description: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"{description} 不存在: {path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{description} 顶层必须是 YAML mapping: {path}")
    return data


def _section(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"配置项 {key} 必须是 YAML mapping")
    return value


def _reject_unknown_keys(config: Dict[str, Any], allowed: set, *, path: str) -> None:
    unknown = sorted(str(key) for key in config if key not in allowed)
    if unknown:
        raise ValueError(f"{path} 包含未知配置项: {unknown}")


def _as_float3(values: Sequence[Any], *, name: str) -> np.ndarray:
    if len(values) != 3:
        raise ValueError(f"{name} 必须是长度为 3 的数组")
    out: List[float] = []
    for idx, value in enumerate(values):
        if value is None:
            raise ValueError(f"{name}[{idx}] 不能为 null，请在 YAML 中显式填写数值")
        out.append(float(value))
    return np.asarray(out, dtype=np.float32)


def _as_optional_bounds(values: Optional[Sequence[Any]], *, name: str) -> Optional[Tuple[float, float]]:
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError(f"{name} 必须是长度为 2 的数组")
    lower = float(values[0])
    upper = float(values[1])
    if lower > upper:
        raise ValueError(f"{name} 下界不能大于上界")
    return lower, upper


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "on"):
            return True
        if normalized in ("false", "0", "no", "off", ""):
            return False
    return bool(value)


def _as_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "auto", "default"):
        return None
    return _as_bool(value)


def _resolve_config_path(path: str, base_dir: Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(base_dir / candidate)


def _stamp_to_sec(stamp: Any) -> Optional[float]:
    if stamp is None:
        return None
    if hasattr(stamp, "to_sec"):
        stamp_sec = float(stamp.to_sec())
        return stamp_sec if stamp_sec > 0.0 else None
    secs = getattr(stamp, "secs", None)
    nsecs = getattr(stamp, "nsecs", None)
    if secs is None or nsecs is None:
        return None
    stamp_sec = float(secs) + float(nsecs) * 1e-9
    return stamp_sec if stamp_sec > 0.0 else None


def _point_to_np(point: Any) -> np.ndarray:
    return np.asarray([float(point.x), float(point.y), float(point.z)], dtype=np.float32)


def _vector_to_np(vector: Any) -> np.ndarray:
    return np.asarray([float(vector.x), float(vector.y), float(vector.z)], dtype=np.float32)


def _state_from_odometry(msg: Any) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    return (
        _point_to_np(msg.pose.pose.position),
        _vector_to_np(msg.twist.twist.linear),
        _stamp_to_sec(getattr(msg.header, "stamp", None)),
    )


def _state_from_point_stamped(
    msg: Any,
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    position = _point_to_np(msg.point)
    stamp_sec = _stamp_to_sec(getattr(msg.header, "stamp", None))
    velocity = np.zeros((3,), dtype=np.float32)
    return position, velocity, stamp_sec


def _normalize_uav_name(value: Any, *, default: str = "uav0") -> str:
    name = str(value or "").strip().strip("/")
    return name or default


def _parse_fleet_uavs(value: Any) -> List[str]:
    if isinstance(value, str):
        raw_items = value.replace(";", ",").split(",")
    elif isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raise ValueError("~fleet_uavs 必须是逗号分隔字符串或 YAML list")

    names: List[str] = []
    for item in raw_items:
        name = _normalize_uav_name(item, default="")
        if name and name not in names:
            names.append(name)
    return names


def _parse_defender_roles(value: Any, *, name: str) -> Tuple[str, ...]:
    if isinstance(value, str):
        raw_items = value.replace(";", ",").split(",")
    elif isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raise ValueError(f"{name} 必须是逗号分隔字符串或 YAML list")

    roles: List[str] = []
    for item in raw_items:
        role = str(item or "").strip()
        if not role:
            continue
        if role not in DEFENDER_ROLES:
            raise ValueError(f"{name} 包含未知 defender role: {role}")
        if role not in roles:
            roles.append(role)
    if not roles:
        raise ValueError(f"{name} 至少需要包含一个 defender role")
    return tuple(roles)


def _topic_to_position_cmd_topic(topic: str) -> str:
    topic = str(topic or "").strip()
    if not topic:
        return ""
    suffix = "/vins_position"
    if topic.endswith(suffix):
        return f"{topic[:-len(suffix)]}/position_cmd"
    return ""


class AdvRosAdapter:
    def __init__(self, model_config_path: str = DEFAULT_MODEL_CONFIG_PATH):
        import rospy
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import PointStamped, PoseStamped
        from quadrotor_msgs.msg import PositionCommand

        self.rospy = rospy
        self.Odometry = Odometry
        self.PointStamped = PointStamped
        self.PoseStamped = PoseStamped
        self.PositionCommand = PositionCommand

        self.model_config_path = model_config_path
        self.model_config = _load_yaml(model_config_path)
        _reject_unknown_keys(self.model_config, TOP_LEVEL_KEYS, path="model_config")
        self.model_config_dir = Path(model_config_path).resolve().parent
        self.adapter_config = _section(self.model_config, "adapter")
        _reject_unknown_keys(self.adapter_config, ADAPTER_KEYS, path="adapter")
        self.active_defender_roles = _parse_defender_roles(
            rospy.get_param(
                "~active_defender_roles",
                self.adapter_config.get("active_defender_roles", list(DEFENDER_ROLES)),
            ),
            name="active_defender_roles",
        )
        self.output_default_height = float(self.adapter_config.get("output_default_height", 1.0))
        self.output_velocity_z = float(self.adapter_config.get("output_velocity_z", 0.0))
        self.output_frame = str(self.adapter_config.get("output_frame", "local"))
        if self.output_frame not in ("local", "world"):
            raise ValueError(f"adapter.output_frame 必须是 local 或 world，实际得到 {self.output_frame}")

        self.auto_role_mapping = bool(rospy.get_param("~auto_role_mapping", False))
        self.auto_role_topics = self._auto_role_topics() if self.auto_role_mapping else None
        output_role_param = str(
            rospy.get_param(
                "~output_role",
                "auto" if self.auto_role_mapping else self.adapter_config.get("output_role", "defender_0"),
            )
        )
        self.output_role = self._auto_output_role() if output_role_param == "auto" else output_role_param
        if self.output_role not in DEFENDER_ROLES:
            raise ValueError(f"output_role 必须是 {DEFENDER_ROLES} 之一，实际得到 {self.output_role}")
        if self.output_role not in self.active_defender_roles:
            raise ValueError(
                f"output_role={self.output_role} 不在 active_defender_roles={self.active_defender_roles} 中"
            )
        self.output_index = DEFENDER_ROLES.index(self.output_role)

        inference_config_path = _resolve_config_path(
            str(self.adapter_config.get("inference_config_path", "inference_config.yaml")),
            self.model_config_dir,
        )
        self.engine = InferenceEngine(config_path=inference_config_path)

        self.command_position_dt = float(self.adapter_config.get("command_position_dt", 0.1))
        self.stale_timeout_sec = float(self.adapter_config.get("stale_timeout_sec", 0.5))
        self.estimate_velocity = bool(self.adapter_config.get("estimate_velocity_from_position", True))
        self.output_frame_id = str(self.adapter_config.get("output_frame_id", "world"))
        self.position_command_config = _section(self.model_config, "position_command")
        _reject_unknown_keys(self.position_command_config, POSITION_COMMAND_KEYS, path="position_command")
        self.game_end_command_topic_overrides: Dict[str, str] = {}
        self.game_end_monitor = self._build_game_end_monitor()
        self.game_end_hover_published = False

        self.transforms = self._build_coordinate_transforms()
        self.states = self._build_default_states()
        self.subscribers = self._build_subscribers()
        self.output_topic = str(rospy.get_param("~output_topic", self.adapter_config.get("output_topic", "/adv/position_cmd")))
        self.publisher = rospy.Publisher(self.output_topic, PositionCommand, queue_size=10)
        self.terminal_command_topics = self._build_terminal_command_topics(self.output_topic)
        self.terminal_publishers = self._build_terminal_publishers()
        self.plan_point_enabled = bool(
            rospy.get_param("~plan_point_enabled", self.adapter_config.get("plan_point_enabled", False))
        )
        self.plan_point_topic = str(
            rospy.get_param("~plan_point_topic", self.adapter_config.get("plan_point_topic", "/toplan/single_plan_point"))
        )
        self.plan_point_frame_id = str(
            rospy.get_param("~plan_point_frame_id", self.adapter_config.get("plan_point_frame_id", self.output_frame_id))
        )
        self.plan_point_publisher = (
            rospy.Publisher(self.plan_point_topic, PoseStamped, queue_size=10)
            if self.plan_point_enabled
            else None
        )

        self._log_startup_summary(model_config_path, self.output_topic)

    def _build_game_end_monitor(self) -> GameEndMonitor:
        game_config = _section(self.model_config, "game_end")
        _reject_unknown_keys(game_config, GAME_END_KEYS, path="game_end")

        enabled_param = _as_optional_bool(self.rospy.get_param("~game_end_enabled", "auto"))
        enabled = _as_bool(game_config.get("enabled", False)) if enabled_param is None else enabled_param
        command_topics = game_config.get("command_topics", {})
        if not isinstance(command_topics, dict):
            raise ValueError("game_end.command_topics 必须是 YAML mapping")
        _reject_unknown_keys(command_topics, set(ALL_ROLES), path="game_end.command_topics")
        self.game_end_command_topic_overrides = {
            str(role): str(topic)
            for role, topic in command_topics.items()
            if str(topic)
        }

        configured_command_topics = self.rospy.get_param("~game_end_command_topics", None)
        if configured_command_topics is not None:
            if not isinstance(configured_command_topics, dict):
                raise ValueError("~game_end_command_topics 必须是 YAML mapping")
            _reject_unknown_keys(configured_command_topics, set(ALL_ROLES), path="~game_end_command_topics")
            self.game_end_command_topic_overrides.update(
                {
                    str(role): str(topic)
                    for role, topic in configured_command_topics.items()
                    if str(topic)
                }
            )

        if not enabled:
            return GameEndMonitor(GameEndConfig(enabled=False))

        inference_config = _load_yaml_mapping(self.engine.config.config_path, description="inference_config")
        safety_bounds_config = _section(_section(inference_config, "safety"), "bounds")

        bounds_config = game_config.get("world_bounds", {})
        if not isinstance(bounds_config, dict):
            raise ValueError("game_end.world_bounds 必须是 YAML mapping")
        _reject_unknown_keys(bounds_config, WORLD_BOUNDS_KEYS, path="game_end.world_bounds")

        asset_origin = game_config.get("asset_origin", self.engine.config.origin_array)
        x_bounds = _as_optional_bounds(bounds_config.get("x"), name="game_end.world_bounds.x")
        y_bounds = _as_optional_bounds(bounds_config.get("y"), name="game_end.world_bounds.y")
        z_bounds = _as_optional_bounds(bounds_config.get("z"), name="game_end.world_bounds.z")
        if x_bounds is None:
            x_bounds = _as_optional_bounds(safety_bounds_config.get("x"), name="inference_config.safety.bounds.x")
        if y_bounds is None:
            y_bounds = _as_optional_bounds(safety_bounds_config.get("y"), name="inference_config.safety.bounds.y")
        if z_bounds is None:
            z_bounds = _as_optional_bounds(safety_bounds_config.get("z"), name="inference_config.safety.bounds.z")

        return GameEndMonitor(
            GameEndConfig(
                enabled=True,
                capture_distance_m=float(game_config.get("capture_distance_m", 1.0)),
                asset_distance_m=float(game_config.get("asset_distance_m", 1.0)),
                hold_duration_sec=float(game_config.get("hold_duration_sec", 0.5)),
                asset_origin=_as_float3(asset_origin, name="game_end.asset_origin"),
                x_bounds=x_bounds,
                y_bounds=y_bounds,
                z_bounds=z_bounds,
                out_of_bounds_enabled=_as_bool(game_config.get("out_of_bounds_enabled", True)),
            )
        )

    def _log_startup_summary(self, model_config_path: str, output_topic: str) -> None:
        self.rospy.loginfo("adv ROS adapter started")
        self.rospy.loginfo("model_config: %s", model_config_path)
        self.rospy.loginfo(
            "output: role=%s topic=%s output_frame=%s output_frame_id=%s",
            self.output_role,
            self.rospy.resolve_name(output_topic),
            self.output_frame,
            self.output_frame_id,
        )
        self.rospy.loginfo("active defender roles: %s", list(self.active_defender_roles))
        if self.game_end_monitor.config.enabled:
            cfg = self.game_end_monitor.config
            self.rospy.loginfo(
                "game end: enabled capture<%.3fm for %.3fs asset<%.3fm for %.3fs asset_origin=%s bounds(x=%s,y=%s,z=%s)",
                cfg.capture_distance_m,
                cfg.hold_duration_sec,
                cfg.asset_distance_m,
                cfg.hold_duration_sec,
                [round(float(value), 6) for value in cfg.asset_origin.tolist()],
                cfg.x_bounds,
                cfg.y_bounds,
                cfg.z_bounds,
            )
            for role in ALL_ROLES:
                topic = self.terminal_command_topics.get(role, "")
                if topic:
                    self.rospy.loginfo("terminal hover output: role=%s topic=%s", role, self.rospy.resolve_name(topic))
        self._warn_if_output_frame_id_mismatch()
        if self.plan_point_enabled:
            self.rospy.loginfo(
                "plan point: topic=%s frame_id=%s",
                self.rospy.resolve_name(self.plan_point_topic),
                self.plan_point_frame_id,
            )
            if self.plan_point_frame_id != self.output_frame_id:
                self.rospy.logwarn(
                    "plan_point_frame_id (%s) 与 output_frame_id (%s) 不一致；"
                    "plan_point 复用 PositionCommand 坐标，通常应保持一致",
                    self.plan_point_frame_id,
                    self.output_frame_id,
                )

        for role in ALL_ROLES:
            topic, msg_type = self.input_descriptions.get(role, ("", "defaults.states"))
            topic_text = self.rospy.resolve_name(topic) if topic else "defaults.states"
            transform = self.transforms[role]
            translation = [round(float(value), 6) for value in transform.translation.tolist()]
            self.rospy.loginfo(
                "role config: role=%s topic=%s type=%s local_to_world.translation=%s yaw_deg=%.3f yaw_rad=%.6f",
                role,
                topic_text,
                msg_type,
                translation,
                math.degrees(transform.yaw_rad),
                transform.yaw_rad,
            )

    def _warn_if_output_frame_id_mismatch(self) -> None:
        if self.output_frame == "local" and self.output_frame_id != self.output_role:
            self.rospy.logwarn(
                "output_frame=local 且 output_role=%s，但 output_frame_id=%s；"
                "通常 output_frame_id 应与 output_role 一致",
                self.output_role,
                self.output_frame_id,
            )
        elif self.output_frame == "world" and self.output_frame_id == self.output_role:
            self.rospy.logwarn(
                "output_frame=world，但 output_frame_id=%s 看起来像 role 本地坐标系；"
                "请确认 header.frame_id 是否应为 world",
                self.output_frame_id,
            )

    def _build_coordinate_transforms(self) -> Dict[str, CoordinateTransform]:
        transforms_config = _section(self.model_config, "coordinate_transforms")
        _reject_unknown_keys(transforms_config, set(ALL_ROLES), path="coordinate_transforms")
        transforms: Dict[str, CoordinateTransform] = {}
        for role in ALL_ROLES:
            role_config = transforms_config.get(role)
            if not isinstance(role_config, dict):
                raise ValueError(f"coordinate_transforms.{role} 必须配置 translation 和 yaw_deg")
            _reject_unknown_keys(role_config, TRANSFORM_KEYS, path=f"coordinate_transforms.{role}")
            transforms[role] = CoordinateTransform(
                translation=_as_float3(role_config.get("translation", [0.0, 0.0, 0.0]), name=f"{role}.translation"),
                yaw_rad=math.radians(float(role_config.get("yaw_deg", 0.0))),
            )
        return transforms

    def _build_default_states(self) -> Dict[str, TopicState]:
        defaults_config = _section(self.model_config, "defaults")
        _reject_unknown_keys(defaults_config, {"states"}, path="defaults")
        states_config = _section(defaults_config, "states")
        _reject_unknown_keys(states_config, set(ALL_ROLES), path="defaults.states")
        states: Dict[str, TopicState] = {}
        for role in ALL_ROLES:
            role_config = states_config.get(role)
            if not isinstance(role_config, dict):
                raise ValueError(f"defaults.states.{role} 必须配置默认 position 和 velocity")
            _reject_unknown_keys(role_config, STATE_KEYS, path=f"defaults.states.{role}")
            local_position = _as_float3(
                role_config.get("position", [0.0, 0.0, None]),
                name=f"{role}.position",
            )
            local_velocity = _as_float3(role_config.get("velocity", [0.0, 0.0, 0.0]), name=f"{role}.velocity")
            transform = self.transforms[role]
            states[role] = TopicState(
                position=transform.local_to_world_position(local_position),
                velocity=transform.local_to_world_velocity(local_velocity),
            )
        return states

    def _build_subscribers(self) -> List[Any]:
        import rospy

        inputs = self.model_config.get("inputs", [])
        if not isinstance(inputs, list):
            raise ValueError("inputs 必须是 YAML list")

        role_topics = self.auto_role_topics
        configured_role_topics = rospy.get_param("~role_topics", None)
        if configured_role_topics is not None:
            role_topics = configured_role_topics
        role_message_types = rospy.get_param("~role_message_types", {})
        if role_topics is not None:
            if not isinstance(role_topics, dict):
                raise ValueError("~role_topics 必须是 YAML mapping")
            if not isinstance(role_message_types, dict):
                raise ValueError("~role_message_types 必须是 YAML mapping")
            inputs = [
                {
                    "role": str(role),
                    "topic": str(topic),
                    "message_type": str(role_message_types.get(role, "auto")),
                }
                for role, topic in role_topics.items()
                if str(topic)
            ]

        subscribers: List[Any] = []
        self.input_descriptions: Dict[str, Tuple[str, str]] = {}
        seen_roles = set()
        for entry in inputs:
            if not isinstance(entry, dict):
                raise ValueError("inputs 中的每一项都必须是 YAML mapping")
            _reject_unknown_keys(entry, INPUT_KEYS, path="inputs[]")
            role = str(entry.get("role", ""))
            topic = str(entry.get("topic", ""))
            msg_type = str(entry.get("message_type", "auto"))
            if role not in ALL_ROLES:
                raise ValueError(f"未知输入角色: {role}")
            if role in DEFENDER_ROLES and role not in self.active_defender_roles:
                continue
            if not topic:
                raise ValueError(f"输入角色 {role} 缺少 topic")
            seen_roles.add(role)
            self.input_descriptions[role] = (topic, msg_type)

            if msg_type == "nav_msgs/Odometry":
                subscribers.append(rospy.Subscriber(topic, self.Odometry, self._make_typed_callback(role, "odometry"), queue_size=10))
            elif msg_type == "geometry_msgs/PointStamped":
                subscribers.append(rospy.Subscriber(topic, self.PointStamped, self._make_typed_callback(role, "point"), queue_size=10))
            elif msg_type == "auto":
                subscribers.append(rospy.Subscriber(topic, rospy.AnyMsg, self._make_any_callback(role), queue_size=10))
            else:
                raise ValueError(f"不支持的 message_type: {msg_type}")

        expected_roles = tuple(self.active_defender_roles) + ("enemy",)
        missing = [role for role in expected_roles if role not in seen_roles]
        if missing:
            rospy.logwarn("inputs 缺少这些角色，将使用 defaults.states: %s", missing)
        return subscribers

    def _build_terminal_command_topics(self, output_topic: str) -> Dict[str, str]:
        topics: Dict[str, str] = {}
        for role, (state_topic, _) in self.input_descriptions.items():
            command_topic = _topic_to_position_cmd_topic(state_topic)
            if command_topic:
                topics[role] = command_topic

        topics.update(self.game_end_command_topic_overrides)
        topics[self.output_role] = output_topic
        return {role: topic for role, topic in topics.items() if role in ALL_ROLES and topic}

    def _build_terminal_publishers(self) -> Dict[str, Any]:
        publishers: Dict[str, Any] = {}
        topic_publishers: Dict[str, Any] = {self.output_topic: self.publisher}
        for role, topic in self.terminal_command_topics.items():
            publisher = topic_publishers.get(topic)
            if publisher is None:
                publisher = self.rospy.Publisher(topic, self.PositionCommand, queue_size=10)
                topic_publishers[topic] = publisher
            publishers[role] = publisher
        return publishers

    def _auto_output_role(self) -> str:
        self_uav = _normalize_uav_name(
            self.rospy.get_param("~self_uav", self.rospy.get_namespace()),
            default="uav0",
        )
        expected_topic = f"/{self_uav}/vins_position"
        role_topics = self.auto_role_topics or {}
        for role in self.active_defender_roles:
            if role_topics.get(role) == expected_topic:
                return role
        raise ValueError(
            f"auto output_role 无法为 {self_uav} 找到 defender 映射；"
            "请检查 fleet_uavs/enemy_uav，或显式传 output_role:=defender_0..3"
        )

    def _auto_role_topics(self) -> Dict[str, str]:
        enemy_uav = _normalize_uav_name(self.rospy.get_param("~enemy_uav", ""), default="")
        fleet_uavs = _parse_fleet_uavs(self.rospy.get_param("~fleet_uavs", "uav0,uav1,uav2"))
        primary_defender_uavs = fleet_uavs[: len(DEFENDER_ROLES)]
        replacement_uavs = [
            name
            for name in fleet_uavs
            if name not in primary_defender_uavs and name != enemy_uav
        ]
        role_topics: Dict[str, str] = {
            "enemy": f"/{enemy_uav}/vins_position" if enemy_uav else "",
        }

        for idx, role in enumerate(DEFENDER_ROLES):
            if role not in self.active_defender_roles:
                role_topics[role] = ""
                continue
            defender_uav = primary_defender_uavs[idx] if idx < len(primary_defender_uavs) else ""
            if defender_uav and defender_uav == enemy_uav:
                defender_uav = replacement_uavs.pop(0) if replacement_uavs else ""
            if defender_uav:
                role_topics[role] = f"/{defender_uav}/vins_position"
            else:
                role_topics[role] = ""

        self.rospy.loginfo(
            "auto role mapping enemy=%s fleet=%s topics=%s",
            enemy_uav or "<default-state>",
            fleet_uavs,
            role_topics,
        )
        return role_topics

    def _make_typed_callback(self, role: str, kind: str):
        def callback(msg: Any) -> None:
            self._update_state(role, msg, kind)

        return callback

    def _make_any_callback(self, role: str):
        def callback(any_msg: Any) -> None:
            msg_type = any_msg._connection_header.get("type", "")
            if msg_type == "nav_msgs/Odometry":
                msg = self.Odometry().deserialize(any_msg._buff)
                self._update_state(role, msg, "odometry")
            elif msg_type == "geometry_msgs/PointStamped":
                msg = self.PointStamped().deserialize(any_msg._buff)
                self._update_state(role, msg, "point")
            else:
                self.rospy.logwarn_throttle(2.0, "unsupported message type on %s: %s", role, msg_type)

        return callback

    def _update_state(self, role: str, msg: Any, kind: str) -> None:
        previous = self.states.get(role)
        if kind == "odometry":
            local_position, local_velocity, stamp_sec = _state_from_odometry(msg)
        elif kind == "point":
            local_position, local_velocity, stamp_sec = _state_from_point_stamped(msg)
        else:
            raise ValueError(f"未知消息类型: {kind}")

        if stamp_sec is None:
            stamp_sec = float(self.rospy.Time.now().to_sec())

        transform = self.transforms[role]
        position = transform.local_to_world_position(local_position)
        velocity = transform.local_to_world_velocity(local_velocity)
        if kind == "point" and self.estimate_velocity and previous is not None and previous.received and previous.stamp_sec is not None:
            dt = float(stamp_sec - previous.stamp_sec)
            if dt > 1e-6:
                velocity = ((position - previous.position) / dt).astype(np.float32)

        self.states[role] = TopicState(
            position=position,
            velocity=velocity,
            stamp_sec=stamp_sec,
            received=True,
        )

    def _state_for_role(self, role: str, now_sec: float) -> TopicState:
        state = self.states[role]
        if not state.received or state.stamp_sec is None:
            return state
        if self.stale_timeout_sec <= 0.0:
            return state
        age = now_sec - state.stamp_sec
        if age > self.stale_timeout_sec:
            self.rospy.logwarn_throttle(2.0, "%s 输入超时 %.3fs，继续使用最近一次/默认状态", role, age)
        return state

    def build_snapshot(self) -> InferenceSnapshot:
        now_sec = float(self.rospy.Time.now().to_sec())
        defenders = [
            VehicleState(
                position=self._state_for_role(role, now_sec).position,
                velocity=self._state_for_role(role, now_sec).velocity,
            )
            for role in DEFENDER_ROLES
        ]
        enemy_state = self._state_for_role("enemy", now_sec)
        return InferenceSnapshot(
            defenders=defenders,
            enemy=VehicleState(position=enemy_state.position, velocity=enemy_state.velocity),
            step_count=int(self.rospy.get_time() * float(self.adapter_config.get("publish_rate_hz", 20.0))),
            active_defender_indices=tuple(DEFENDER_ROLES.index(role) for role in self.active_defender_roles),
        )

    def _received_world_positions(self, now_sec: float) -> Dict[str, np.ndarray]:
        positions: Dict[str, np.ndarray] = {}
        for role in ALL_ROLES:
            state = self._state_for_role(role, now_sec)
            if state.received:
                positions[role] = state.position.copy()
        return positions

    def update_game_end(self, now_sec: Optional[float] = None) -> GameEndStatus:
        if now_sec is None:
            now_sec = float(self.rospy.Time.now().to_sec())
        return self.game_end_monitor.update(self._received_world_positions(now_sec), float(now_sec))

    def _role_output_frame_id(self, role: str) -> str:
        if self.output_frame == "local" and self.output_frame_id == self.output_role:
            return role
        return self.output_frame_id

    def _convert_world_command_for_role(
        self,
        role: str,
        position_world: np.ndarray,
        velocity_world: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.output_frame == "local":
            output_transform = self.transforms[role]
            return (
                output_transform.world_to_local_position(position_world),
                output_transform.world_to_local_velocity(velocity_world),
            )
        return position_world.astype(np.float32), velocity_world.astype(np.float32)

    def _make_position_command(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        *,
        frame_id: str,
    ):
        msg = self.PositionCommand()
        msg.header.stamp = self.rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.position.x = float(position[0])
        msg.position.y = float(position[1])
        msg.position.z = float(position[2])
        msg.velocity.x = float(velocity[0])
        msg.velocity.y = float(velocity[1])
        msg.velocity.z = float(velocity[2])
        msg.acceleration.x = 0.0
        msg.acceleration.y = 0.0
        msg.acceleration.z = 0.0
        msg.jerk.x = 0.0
        msg.jerk.y = 0.0
        msg.jerk.z = 0.0
        msg.yaw = float(self.position_command_config.get("yaw", 0.0))
        msg.yaw_dot = float(self.position_command_config.get("yaw_dot", 0.0))
        msg.kx = [float(v) for v in self.position_command_config.get("kx", [5.7, 5.7, 6.2])]
        msg.kv = [float(v) for v in self.position_command_config.get("kv", [3.4, 3.4, 4.0])]
        msg.trajectory_id = int(self.position_command_config.get("trajectory_id", 1))
        msg.trajectory_flag = self.PositionCommand.TRAJECTORY_STATUS_READY
        return msg

    def build_outputs(self, snapshot: InferenceSnapshot):
        result = self.engine.predict(snapshot)
        command = result.commands[self.output_index]
        defender_state = snapshot.defenders[self.output_index]
        target_position_world = defender_state.position + command.velocity_xyz * self.command_position_dt
        target_velocity_world = command.velocity_xyz.copy()
        target_position, target_velocity = self._convert_world_command_for_role(
            self.output_role,
            target_position_world,
            target_velocity_world,
        )

        target_position[2] = float(self.output_default_height)
        target_velocity[2] = float(self.output_velocity_z)

        msg = self._make_position_command(
            target_position,
            target_velocity,
            frame_id=self.output_frame_id,
        )

        plan_point = None
        if self.plan_point_enabled:
            plan_point = self.PoseStamped()
            plan_point.header.stamp = msg.header.stamp
            plan_point.header.frame_id = self.plan_point_frame_id
            plan_point.pose.position.x = msg.position.x
            plan_point.pose.position.y = msg.position.y
            plan_point.pose.position.z = msg.position.z
            plan_point.pose.orientation.w = 1.0

        return msg, plan_point

    def publish_terminal_hover_once(self, status: GameEndStatus) -> None:
        if self.game_end_hover_published:
            return

        now_sec = float(self.rospy.Time.now().to_sec())
        missing_roles: List[str] = []
        published_roles: List[str] = []
        for role in ALL_ROLES:
            publisher = self.terminal_publishers.get(role)
            if publisher is None:
                missing_roles.append(role)
                continue

            state = self._state_for_role(role, now_sec)
            if not state.received:
                missing_roles.append(role)
                continue

            position, velocity = self._convert_world_command_for_role(
                role,
                state.position.copy(),
                np.zeros((3,), dtype=np.float32),
            )
            msg = self._make_position_command(
                position,
                velocity,
                frame_id=self._role_output_frame_id(role),
            )
            publisher.publish(msg)
            published_roles.append(role)

        self.game_end_hover_published = True
        self.rospy.logwarn(
            "ADV game ended: outcome=%s reason=%s trigger_roles=%s trigger_distance=%s. "
            "Published terminal hover for roles=%s",
            status.outcome,
            status.reason,
            list(status.trigger_roles),
            "n/a" if status.trigger_distance_m is None else f"{status.trigger_distance_m:.3f}",
            published_roles,
        )
        if missing_roles:
            self.rospy.logwarn("terminal hover skipped roles without publisher or received state: %s", missing_roles)

    def spin(self) -> None:
        rate_hz = float(self.adapter_config.get("publish_rate_hz", 20.0))
        rate = self.rospy.Rate(rate_hz)
        while not self.rospy.is_shutdown():
            try:
                game_end_status = self.update_game_end()
                if game_end_status.active:
                    self.publish_terminal_hover_once(game_end_status)
                    rate.sleep()
                    continue

                position_command, plan_point = self.build_outputs(self.build_snapshot())
                self.publisher.publish(position_command)
                if plan_point is not None and self.plan_point_publisher is not None:
                    self.plan_point_publisher.publish(plan_point)
            except Exception as exc:
                self.rospy.logerr_throttle(1.0, "adv adapter publish failed: %s", exc)
            rate.sleep()


def main() -> None:
    import rospy

    rospy.init_node("adv_inference_adapter")
    model_config_path = rospy.get_param("~model_config", DEFAULT_MODEL_CONFIG_PATH)
    AdvRosAdapter(model_config_path=model_config_path).spin()


if __name__ == "__main__":
    main()
