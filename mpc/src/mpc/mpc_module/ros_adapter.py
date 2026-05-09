#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from .mpc_engine import MpcEngine, MpcSnapshot, VehicleState


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_CONFIG_PATH = str(MODULE_DIR / "model_config.yaml")
ENEMY_ROLE = "enemy"
DEFENDER_ROLES = ("defender_0", "defender_1", "defender_2", "defender_3")
ALL_ROLES = (ENEMY_ROLE,) + DEFENDER_ROLES


@dataclass
class TopicState:
    position: np.ndarray
    velocity: np.ndarray
    stamp_sec: Optional[float] = None
    received: bool = False


@dataclass
class CoordinateTransform:
    translation: np.ndarray
    yaw_rad: float

    @property
    def rotation(self) -> np.ndarray:
        c = math.cos(self.yaw_rad)
        s = math.sin(self.yaw_rad)
        return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    def local_to_world_position(self, position: np.ndarray) -> np.ndarray:
        return (self.rotation.dot(position.astype(np.float32)) + self.translation).astype(np.float32)

    def local_to_world_velocity(self, velocity: np.ndarray) -> np.ndarray:
        return self.rotation.dot(velocity.astype(np.float32)).astype(np.float32)

    def world_to_local_position(self, position: np.ndarray) -> np.ndarray:
        return self.rotation.T.dot(position.astype(np.float32) - self.translation).astype(np.float32)

    def world_to_local_velocity(self, velocity: np.ndarray) -> np.ndarray:
        return self.rotation.T.dot(velocity.astype(np.float32)).astype(np.float32)


def _load_yaml(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"model_config 不存在: {path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"model_config 顶层必须是 YAML mapping: {path}")
    return data


def _section(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"配置项 {key} 必须是 YAML mapping")
    return value


def _as_float3(values: Sequence[Any], *, name: str) -> np.ndarray:
    if len(values) != 3:
        raise ValueError(f"{name} 必须是长度为 3 的数组")
    out: List[float] = []
    for idx, value in enumerate(values):
        if value is None:
            raise ValueError(f"{name}[{idx}] 不能为 null，请在 YAML 中显式填写数值")
        out.append(float(value))
    return np.asarray(out, dtype=np.float32)


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
    return _point_to_np(msg.pose.pose.position), _vector_to_np(msg.twist.twist.linear), _stamp_to_sec(msg.header.stamp)


def _state_from_point_stamped(msg: Any) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    return _point_to_np(msg.point), np.zeros((3,), dtype=np.float32), _stamp_to_sec(msg.header.stamp)


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


class MpcRosAdapter:
    def __init__(self, model_config_path: str = DEFAULT_MODEL_CONFIG_PATH):
        import rospy
        from geometry_msgs.msg import PointStamped, PoseStamped
        from nav_msgs.msg import Odometry
        from quadrotor_msgs.msg import PositionCommand

        self.rospy = rospy
        self.PointStamped = PointStamped
        self.PoseStamped = PoseStamped
        self.Odometry = Odometry
        self.PositionCommand = PositionCommand

        self.model_config_path = model_config_path
        self.model_config = _load_yaml(model_config_path)
        self.model_config_dir = Path(model_config_path).resolve().parent
        self.adapter_config = _section(self.model_config, "adapter")

        self.output_frame = str(self.adapter_config.get("output_frame", "local"))
        if self.output_frame not in ("local", "world"):
            raise ValueError(f"adapter.output_frame 必须是 local 或 world，实际得到 {self.output_frame}")
        self.output_frame_role = str(self.adapter_config.get("output_frame_role", ENEMY_ROLE))
        if self.output_frame_role not in ALL_ROLES:
            raise ValueError(f"未知 output_frame_role: {self.output_frame_role}")
        self.output_frame_id = str(self.adapter_config.get("output_frame_id", self.output_frame_role))
        self.output_default_height = float(self.adapter_config.get("output_default_height", 1.0))
        self.output_velocity_z = float(self.adapter_config.get("output_velocity_z", 0.0))
        self.publish_rate_hz = float(self.adapter_config.get("publish_rate_hz", 20.0))
        self.stale_timeout_sec = float(self.adapter_config.get("stale_timeout_sec", 0.5))
        self.position_command_config = _section(self.model_config, "position_command")

        mpc_config_path = _resolve_config_path(str(self.adapter_config.get("mpc_config_path", "mpc_config.yaml")), self.model_config_dir)
        self.engine = MpcEngine(config_path=mpc_config_path)

        self.transforms = self._build_coordinate_transforms()
        self.states = self._build_default_states()
        self.subscribers = self._build_subscribers()

        output_topic = str(rospy.get_param("~output_topic", self.adapter_config.get("output_topic", "/uav5/position_cmd")))
        self.publisher = rospy.Publisher(output_topic, PositionCommand, queue_size=10)
        self.plan_point_enabled = bool(
            rospy.get_param("~plan_point_enabled", self.adapter_config.get("plan_point_enabled", True))
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
        rospy.loginfo("mpc ROS adapter started")
        rospy.loginfo("model_config: %s", model_config_path)
        rospy.loginfo("output topic/frame: %s (%s)", rospy.resolve_name(output_topic), self.output_frame)
        if self.plan_point_enabled:
            rospy.loginfo("plan point topic: %s", rospy.resolve_name(self.plan_point_topic))

    def _build_coordinate_transforms(self) -> Dict[str, CoordinateTransform]:
        transforms_config = _section(self.model_config, "coordinate_transforms")
        transforms: Dict[str, CoordinateTransform] = {}
        for role in ALL_ROLES:
            role_config = transforms_config.get(role)
            if not isinstance(role_config, dict):
                raise ValueError(f"coordinate_transforms.{role} 必须配置 translation 和 yaw_deg")
            transforms[role] = CoordinateTransform(
                translation=_as_float3(role_config.get("translation", [0.0, 0.0, 0.0]), name=f"{role}.translation"),
                yaw_rad=math.radians(float(role_config.get("yaw_deg", 0.0))),
            )
        return transforms

    def _build_default_states(self) -> Dict[str, TopicState]:
        states_config = _section(_section(self.model_config, "defaults"), "states")
        states: Dict[str, TopicState] = {}
        for role in ALL_ROLES:
            role_config = states_config.get(role)
            if not isinstance(role_config, dict):
                raise ValueError(f"defaults.states.{role} 必须配置默认 position 和 velocity")
            local_position = _as_float3(role_config.get("position", [0.0, 0.0, 1.0]), name=f"{role}.position")
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

        role_topics = self._auto_role_topics() if bool(rospy.get_param("~auto_role_mapping", False)) else None
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
        seen_roles = set()
        for entry in inputs:
            if not isinstance(entry, dict):
                raise ValueError("inputs 中每一项都必须是 YAML mapping")
            role = str(entry.get("role", ""))
            topic = str(entry.get("topic", ""))
            msg_type = str(entry.get("message_type", "auto"))
            if role not in ALL_ROLES:
                raise ValueError(f"未知输入角色: {role}")
            if not topic:
                raise ValueError(f"输入角色 {role} 缺少 topic")
            seen_roles.add(role)
            if msg_type == "nav_msgs/Odometry":
                subscribers.append(rospy.Subscriber(topic, self.Odometry, self._make_typed_callback(role, "odometry"), queue_size=10))
            elif msg_type == "geometry_msgs/PointStamped":
                subscribers.append(rospy.Subscriber(topic, self.PointStamped, self._make_typed_callback(role, "point"), queue_size=10))
            elif msg_type == "auto":
                subscribers.append(rospy.Subscriber(topic, rospy.AnyMsg, self._make_any_callback(role), queue_size=10))
            else:
                raise ValueError(f"不支持的 message_type: {msg_type}")
            rospy.loginfo("input role/topic/type: %s <- %s (%s)", role, rospy.resolve_name(topic), msg_type)
        missing = [role for role in ALL_ROLES if role not in seen_roles]
        if missing:
            rospy.logwarn("inputs 缺少这些角色，将使用 defaults.states: %s", missing)
        return subscribers

    def _auto_role_topics(self) -> Dict[str, str]:
        self_uav = _normalize_uav_name(
            self.rospy.get_param("~self_uav", self.rospy.get_namespace()),
            default="uav0",
        )
        fleet_uavs = _parse_fleet_uavs(self.rospy.get_param("~fleet_uavs", "uav0,uav1,uav2,uav3,uav4"))
        if self_uav not in fleet_uavs:
            fleet_uavs.insert(0, self_uav)

        defender_uavs = [name for name in fleet_uavs if name != self_uav]
        role_topics: Dict[str, str] = {
            ENEMY_ROLE: f"/{self_uav}/vins_position",
        }
        for idx, role in enumerate(DEFENDER_ROLES):
            if idx < len(defender_uavs):
                role_topics[role] = f"/{defender_uavs[idx]}/vins_position"
            else:
                role_topics[role] = ""

        self.rospy.loginfo("auto role mapping self=%s fleet=%s topics=%s", self_uav, fleet_uavs, role_topics)
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
        if kind == "odometry":
            local_position, local_velocity, stamp_sec = _state_from_odometry(msg)
        elif kind == "point":
            local_position, local_velocity, stamp_sec = _state_from_point_stamped(msg)
        else:
            raise ValueError(f"未知消息类型: {kind}")
        if stamp_sec is None:
            stamp_sec = float(self.rospy.Time.now().to_sec())
        transform = self.transforms[role]
        self.states[role] = TopicState(
            position=transform.local_to_world_position(local_position),
            velocity=transform.local_to_world_velocity(local_velocity),
            stamp_sec=stamp_sec,
            received=True,
        )

    def _state_for_role(self, role: str, now_sec: float) -> TopicState:
        state = self.states[role]
        if not state.received or state.stamp_sec is None or self.stale_timeout_sec <= 0.0:
            return state
        age = now_sec - state.stamp_sec
        if age > self.stale_timeout_sec:
            self.rospy.logwarn_throttle(2.0, "%s 输入超时 %.3fs，继续使用最近一次/默认状态", role, age)
        return state

    def build_snapshot(self) -> MpcSnapshot:
        now_sec = float(self.rospy.Time.now().to_sec())
        enemy_state = self._state_for_role(ENEMY_ROLE, now_sec)
        defenders = [
            VehicleState(position=self._state_for_role(role, now_sec).position, velocity=self._state_for_role(role, now_sec).velocity)
            for role in DEFENDER_ROLES
        ]
        return MpcSnapshot(
            enemy=VehicleState(position=enemy_state.position, velocity=enemy_state.velocity),
            defenders=defenders,
            step_count=int(self.rospy.get_time() * self.publish_rate_hz),
        )

    def build_outputs(self, snapshot: MpcSnapshot):
        result = self.engine.plan(snapshot)
        target_world = result.predicted_position.copy()
        velocity_world = result.velocity_xyz.copy()
        target_world[2] = float(self.output_default_height)
        if self.output_frame == "local":
            output_transform = self.transforms[self.output_frame_role]
            target = output_transform.world_to_local_position(target_world)
            target_velocity = output_transform.world_to_local_velocity(velocity_world)
        else:
            target = target_world.astype(np.float32)
            target_velocity = velocity_world.astype(np.float32)

        msg = self.PositionCommand()
        msg.header.stamp = self.rospy.Time.now()
        msg.header.frame_id = self.output_frame_id
        msg.position.x = float(target[0])
        msg.position.y = float(target[1])
        msg.position.z = float(target[2])
        msg.velocity.x = float(target_velocity[0])
        msg.velocity.y = float(target_velocity[1])
        msg.velocity.z = float(self.output_velocity_z)
        msg.acceleration.x = 0.0
        msg.acceleration.y = 0.0
        msg.acceleration.z = 0.0
        msg.jerk.x = 0.0
        msg.jerk.y = 0.0
        msg.jerk.z = 0.0
        msg.yaw = float(self.position_command_config.get("yaw", result.yaw_rad))
        msg.yaw_dot = float(self.position_command_config.get("yaw_dot", 0.0))
        msg.kx = [float(v) for v in self.position_command_config.get("kx", [5.7, 5.7, 6.2])]
        msg.kv = [float(v) for v in self.position_command_config.get("kv", [3.4, 3.4, 4.0])]
        msg.trajectory_id = int(self.position_command_config.get("trajectory_id", 1))
        msg.trajectory_flag = self.PositionCommand.TRAJECTORY_STATUS_READY

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

    def spin(self) -> None:
        rate = self.rospy.Rate(self.publish_rate_hz)
        while not self.rospy.is_shutdown():
            try:
                position_command, plan_point = self.build_outputs(self.build_snapshot())
                self.publisher.publish(position_command)
                if plan_point is not None and self.plan_point_publisher is not None:
                    self.plan_point_publisher.publish(plan_point)
            except Exception as exc:
                self.rospy.logerr_throttle(1.0, "mpc adapter publish failed: %s", exc)
            rate.sleep()


def main() -> None:
    import rospy

    rospy.init_node("mpc_adapter")
    model_config_path = rospy.get_param("~model_config", DEFAULT_MODEL_CONFIG_PATH)
    MpcRosAdapter(model_config_path=model_config_path).spin()


if __name__ == "__main__":
    main()
