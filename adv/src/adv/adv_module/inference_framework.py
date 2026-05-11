#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone inference framework

This module keeps only the policy reasoning path:
1. Load the shared Actor checkpoint.
2. Build role/slot assignments for four defenders.
3. Build 25D observations or accept prebuilt 25D observations.
4. Run Actor inference and postprocess commands.
"""

from __future__ import annotations

import math
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise ImportError(
        "adv_module requires PyTorch. Install torch in the Python environment used to run inference."
    ) from exc


MODULE_DIR = Path(__file__).resolve().parent


def _resolve_default_config_path() -> str:
    return str(MODULE_DIR / "inference_config.yaml")


DEFAULT_CONFIG_PATH = _resolve_default_config_path()


def _load_yaml_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是 YAML mapping: {config_path}")
    return data


def _resolve_path(value: str, base_dir: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(base_dir / path)


def _section(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"配置项 {name} 必须是 YAML mapping")
    return value


def _cfg(config: Dict[str, Any], section_name: str, key: str, default: Any) -> Any:
    return _section(config, section_name).get(key, default)


def _bounds(config: Dict[str, Any], axis: str, default: Tuple[float, float]) -> Tuple[float, float]:
    bounds = _section(config, "safety").get("bounds", {})
    if not isinstance(bounds, dict):
        raise ValueError("配置项 safety.bounds 必须是 YAML mapping")
    value = bounds.get(axis, default)
    if len(value) != 2:
        raise ValueError(f"配置项 safety.bounds.{axis} 必须是长度为 2 的数组")
    return float(value[0]), float(value[1])


_RUNTIME_CONFIG = _load_yaml_config(DEFAULT_CONFIG_PATH)
DEFAULT_CHECKPOINT_PATH = _resolve_path(
    str(_cfg(_RUNTIME_CONFIG, "inference", "checkpoint_path", "policy_checkpoint.pt")),
    MODULE_DIR,
)


ROLE_INTERCEPTOR = 1
ROLE_BLOCKER = 0

ROLE_HOLD_STEPS = int(_cfg(_RUNTIME_CONFIG, "roles", "hold_steps", 5))
ROLE_REPLAN_MIN_STEPS = int(_cfg(_RUNTIME_CONFIG, "roles", "replan_min_steps", 2))
ROLE_BREAK_ENEMY_DIST = float(_cfg(_RUNTIME_CONFIG, "roles", "break_enemy_dist", 26.0))
ROLE_BREAK_SLOT_ERR_SCALE = float(_cfg(_RUNTIME_CONFIG, "roles", "break_slot_err_scale", 0.60))
ROLE_KILL_TRIGGER = float(_cfg(_RUNTIME_CONFIG, "roles", "kill_trigger", 20.0))
ROLE_KILL_LATE_FRAC = float(_cfg(_RUNTIME_CONFIG, "roles", "kill_late_frac", 0.58))
ROLE_KILL_LATE_DIST = float(_cfg(_RUNTIME_CONFIG, "roles", "kill_late_dist", 26.0))
ROLE_KILL_HOLD_STEPS = int(_cfg(_RUNTIME_CONFIG, "roles", "kill_hold_steps", 6))
ROLE_ORBIT_SCALE = float(_cfg(_RUNTIME_CONFIG, "roles", "orbit_scale", 0.78))
ROLE_GUARD_FORWARD_OFFSET = float(_cfg(_RUNTIME_CONFIG, "roles", "guard_forward_offset", 4.2))
ROLE_ATTACK_LAT_MIN = float(_cfg(_RUNTIME_CONFIG, "roles", "attack_lat_min", 2.4))
ROLE_ATTACK_LAT_MAX = float(_cfg(_RUNTIME_CONFIG, "roles", "attack_lat_max", 4.8))
ROLE_SUPPORT_FORWARD = float(_cfg(_RUNTIME_CONFIG, "roles", "support_forward", 2.2))
ROLE_SLOT_OBS_SCALE = float(_cfg(_RUNTIME_CONFIG, "roles", "slot_obs_scale", 24.0))
ROLE_GATE_SLOT_SCALE = float(_cfg(_RUNTIME_CONFIG, "roles", "gate_slot_scale", 7.0))
ROLE_INTERCEPT_SLOT_SCALE = float(_cfg(_RUNTIME_CONFIG, "roles", "intercept_slot_scale", 8.5))

DEF_VMAX = float(_cfg(_RUNTIME_CONFIG, "defender", "vmax", 13.5))
DEF_INTERCEPTOR_SPEED_FLOOR = float(_cfg(_RUNTIME_CONFIG, "defender", "interceptor_speed_floor", 1.00))
DEF_BLOCKER_SPEED_SCALE = float(_cfg(_RUNTIME_CONFIG, "defender", "blocker_speed_scale", 1.00))
DEFENDER_OVERRIDE_BLEND = float(_cfg(_RUNTIME_CONFIG, "defender", "override_blend", 0.48))
DEFENDER_OVERRIDE_BLEND_KILL = float(_cfg(_RUNTIME_CONFIG, "defender", "override_blend_kill", 0.86))
DEFENDER_OVERRIDE_BLEND_GATE = float(_cfg(_RUNTIME_CONFIG, "defender", "override_blend_gate", 0.80))
DEFENDER_OVERSHOOT_BLEND = float(_cfg(_RUNTIME_CONFIG, "defender", "overshoot_blend", 0.94))
DEFENDER_FINISH_DIST = float(_cfg(_RUNTIME_CONFIG, "defender", "finish_dist", 4.8))
DEFENDER_FINISH_EO = float(_cfg(_RUNTIME_CONFIG, "defender", "finish_enemy_origin_dist", 12.0))
DEFENDER_FINISH_BLEND = float(_cfg(_RUNTIME_CONFIG, "defender", "finish_blend", 0.985))
DEFENDER_OVERSHOOT_PROJ = float(_cfg(_RUNTIME_CONFIG, "defender", "overshoot_proj", 0.42))
DEFENDER_OVERSHOOT_LAT = float(_cfg(_RUNTIME_CONFIG, "defender", "overshoot_lat", 0.16))
DEFENDER_TAIL_LAT_MAX = float(_cfg(_RUNTIME_CONFIG, "defender", "tail_lat_max", 0.16))

DEF_NEAR_ENEMY_SLOW_ENABLED = bool(_cfg(_RUNTIME_CONFIG, "enemy_slowdown", "enabled", True))
DEF_SLOW_DIST_NEAR = float(_cfg(_RUNTIME_CONFIG, "enemy_slowdown", "dist_near", 3.5))
DEF_SLOW_DIST_MID = float(_cfg(_RUNTIME_CONFIG, "enemy_slowdown", "dist_mid", 6.5))
DEF_SLOW_DIST_FAR = float(_cfg(_RUNTIME_CONFIG, "enemy_slowdown", "dist_far", 12.0))
DEF_SLOW_SCALE_CLOSE = float(_cfg(_RUNTIME_CONFIG, "enemy_slowdown", "scale_close", 0.28))
DEF_SLOW_SCALE_NEAR = float(_cfg(_RUNTIME_CONFIG, "enemy_slowdown", "scale_near", 0.55))
DEF_SLOW_SCALE_FAR = float(_cfg(_RUNTIME_CONFIG, "enemy_slowdown", "scale_far", 0.92))
DEF_SLOW_LATERAL = float(_cfg(_RUNTIME_CONFIG, "enemy_slowdown", "lateral", 10.0))
DEF_SLOW_MIN_PROJ = float(_cfg(_RUNTIME_CONFIG, "enemy_slowdown", "min_proj", 0.5))
DEF_SLOW_APPROACH_COS = float(_cfg(_RUNTIME_CONFIG, "enemy_slowdown", "approach_cos", 0.45))

REPULSE_RANGE = float(_cfg(_RUNTIME_CONFIG, "safety", "repulse_range", 5.0))
REPULSE_GAIN = float(_cfg(_RUNTIME_CONFIG, "safety", "repulse_gain", 6.5))
BOUNDARY_MARGIN = float(_cfg(_RUNTIME_CONFIG, "safety", "boundary_margin", 22.0))
BOUNDARY_GAIN = float(_cfg(_RUNTIME_CONFIG, "safety", "boundary_gain", 5.0))
X_MIN, X_MAX = _bounds(_RUNTIME_CONFIG, "x", (-95.0, 95.0))
Y_MIN, Y_MAX = _bounds(_RUNTIME_CONFIG, "y", (-95.0, 95.0))
Z_MIN, Z_MAX = _bounds(_RUNTIME_CONFIG, "z", (-40.0, -4.0))


def _apply_runtime_config(config: Dict[str, Any], base_dir: Path) -> None:
    global _RUNTIME_CONFIG, DEFAULT_CHECKPOINT_PATH
    global ROLE_HOLD_STEPS, ROLE_REPLAN_MIN_STEPS, ROLE_BREAK_ENEMY_DIST, ROLE_BREAK_SLOT_ERR_SCALE
    global ROLE_KILL_TRIGGER, ROLE_KILL_LATE_FRAC, ROLE_KILL_LATE_DIST, ROLE_KILL_HOLD_STEPS
    global ROLE_ORBIT_SCALE, ROLE_GUARD_FORWARD_OFFSET, ROLE_ATTACK_LAT_MIN, ROLE_ATTACK_LAT_MAX
    global ROLE_SUPPORT_FORWARD, ROLE_SLOT_OBS_SCALE, ROLE_GATE_SLOT_SCALE, ROLE_INTERCEPT_SLOT_SCALE
    global DEF_VMAX, DEF_INTERCEPTOR_SPEED_FLOOR, DEF_BLOCKER_SPEED_SCALE, DEFENDER_OVERRIDE_BLEND
    global DEFENDER_OVERRIDE_BLEND_KILL, DEFENDER_OVERRIDE_BLEND_GATE, DEFENDER_OVERSHOOT_BLEND
    global DEFENDER_FINISH_DIST, DEFENDER_FINISH_EO, DEFENDER_FINISH_BLEND, DEFENDER_OVERSHOOT_PROJ
    global DEFENDER_OVERSHOOT_LAT, DEFENDER_TAIL_LAT_MAX
    global DEF_NEAR_ENEMY_SLOW_ENABLED, DEF_SLOW_DIST_NEAR, DEF_SLOW_DIST_MID, DEF_SLOW_DIST_FAR
    global DEF_SLOW_SCALE_CLOSE, DEF_SLOW_SCALE_NEAR, DEF_SLOW_SCALE_FAR, DEF_SLOW_LATERAL
    global DEF_SLOW_MIN_PROJ, DEF_SLOW_APPROACH_COS
    global REPULSE_RANGE, REPULSE_GAIN, BOUNDARY_MARGIN, BOUNDARY_GAIN
    global X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX

    _RUNTIME_CONFIG = config
    DEFAULT_CHECKPOINT_PATH = _resolve_path(
        str(_cfg(config, "inference", "checkpoint_path", "policy_checkpoint.pt")),
        base_dir,
    )

    ROLE_HOLD_STEPS = int(_cfg(config, "roles", "hold_steps", 5))
    ROLE_REPLAN_MIN_STEPS = int(_cfg(config, "roles", "replan_min_steps", 2))
    ROLE_BREAK_ENEMY_DIST = float(_cfg(config, "roles", "break_enemy_dist", 26.0))
    ROLE_BREAK_SLOT_ERR_SCALE = float(_cfg(config, "roles", "break_slot_err_scale", 0.60))
    ROLE_KILL_TRIGGER = float(_cfg(config, "roles", "kill_trigger", 20.0))
    ROLE_KILL_LATE_FRAC = float(_cfg(config, "roles", "kill_late_frac", 0.58))
    ROLE_KILL_LATE_DIST = float(_cfg(config, "roles", "kill_late_dist", 26.0))
    ROLE_KILL_HOLD_STEPS = int(_cfg(config, "roles", "kill_hold_steps", 6))
    ROLE_ORBIT_SCALE = float(_cfg(config, "roles", "orbit_scale", 0.78))
    ROLE_GUARD_FORWARD_OFFSET = float(_cfg(config, "roles", "guard_forward_offset", 4.2))
    ROLE_ATTACK_LAT_MIN = float(_cfg(config, "roles", "attack_lat_min", 2.4))
    ROLE_ATTACK_LAT_MAX = float(_cfg(config, "roles", "attack_lat_max", 4.8))
    ROLE_SUPPORT_FORWARD = float(_cfg(config, "roles", "support_forward", 2.2))
    ROLE_SLOT_OBS_SCALE = float(_cfg(config, "roles", "slot_obs_scale", 24.0))
    ROLE_GATE_SLOT_SCALE = float(_cfg(config, "roles", "gate_slot_scale", 7.0))
    ROLE_INTERCEPT_SLOT_SCALE = float(_cfg(config, "roles", "intercept_slot_scale", 8.5))

    DEF_VMAX = float(_cfg(config, "defender", "vmax", 13.5))
    DEF_INTERCEPTOR_SPEED_FLOOR = float(_cfg(config, "defender", "interceptor_speed_floor", 1.00))
    DEF_BLOCKER_SPEED_SCALE = float(_cfg(config, "defender", "blocker_speed_scale", 1.00))
    DEFENDER_OVERRIDE_BLEND = float(_cfg(config, "defender", "override_blend", 0.48))
    DEFENDER_OVERRIDE_BLEND_KILL = float(_cfg(config, "defender", "override_blend_kill", 0.86))
    DEFENDER_OVERRIDE_BLEND_GATE = float(_cfg(config, "defender", "override_blend_gate", 0.80))
    DEFENDER_OVERSHOOT_BLEND = float(_cfg(config, "defender", "overshoot_blend", 0.94))
    DEFENDER_FINISH_DIST = float(_cfg(config, "defender", "finish_dist", 4.8))
    DEFENDER_FINISH_EO = float(_cfg(config, "defender", "finish_enemy_origin_dist", 12.0))
    DEFENDER_FINISH_BLEND = float(_cfg(config, "defender", "finish_blend", 0.985))
    DEFENDER_OVERSHOOT_PROJ = float(_cfg(config, "defender", "overshoot_proj", 0.42))
    DEFENDER_OVERSHOOT_LAT = float(_cfg(config, "defender", "overshoot_lat", 0.16))
    DEFENDER_TAIL_LAT_MAX = float(_cfg(config, "defender", "tail_lat_max", 0.16))

    DEF_NEAR_ENEMY_SLOW_ENABLED = bool(_cfg(config, "enemy_slowdown", "enabled", True))
    DEF_SLOW_DIST_NEAR = float(_cfg(config, "enemy_slowdown", "dist_near", 3.5))
    DEF_SLOW_DIST_MID = float(_cfg(config, "enemy_slowdown", "dist_mid", 6.5))
    DEF_SLOW_DIST_FAR = float(_cfg(config, "enemy_slowdown", "dist_far", 12.0))
    DEF_SLOW_SCALE_CLOSE = float(_cfg(config, "enemy_slowdown", "scale_close", 0.28))
    DEF_SLOW_SCALE_NEAR = float(_cfg(config, "enemy_slowdown", "scale_near", 0.55))
    DEF_SLOW_SCALE_FAR = float(_cfg(config, "enemy_slowdown", "scale_far", 0.92))
    DEF_SLOW_LATERAL = float(_cfg(config, "enemy_slowdown", "lateral", 10.0))
    DEF_SLOW_MIN_PROJ = float(_cfg(config, "enemy_slowdown", "min_proj", 0.5))
    DEF_SLOW_APPROACH_COS = float(_cfg(config, "enemy_slowdown", "approach_cos", 0.45))

    REPULSE_RANGE = float(_cfg(config, "safety", "repulse_range", 5.0))
    REPULSE_GAIN = float(_cfg(config, "safety", "repulse_gain", 6.5))
    BOUNDARY_MARGIN = float(_cfg(config, "safety", "boundary_margin", 22.0))
    BOUNDARY_GAIN = float(_cfg(config, "safety", "boundary_gain", 5.0))
    X_MIN, X_MAX = _bounds(config, "x", (-95.0, 95.0))
    Y_MIN, Y_MAX = _bounds(config, "y", (-95.0, 95.0))
    Z_MIN, Z_MAX = _bounds(config, "z", (-40.0, -4.0))


def _as_np3(values: Sequence[float], *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError(f"{name} 必须是长度为 3 的向量，实际得到 {arr.shape}")
    return arr


def clip_vec2(vx: float, vy: float, vmax: float) -> Tuple[float, float]:
    speed = float(np.hypot(vx, vy))
    if speed > vmax:
        scale = vmax / (speed + 1e-9)
        return vx * scale, vy * scale
    return vx, vy


def wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def vec2_to_yaw(v: Sequence[float]) -> float:
    return float(np.arctan2(float(v[1]), float(v[0])))


def yaw_to_unit(yaw: float) -> np.ndarray:
    return np.array([math.cos(float(yaw)), math.sin(float(yaw))], dtype=np.float32)


def defender_speed_scale_by_enemy_dist(
    def_xy: Sequence[float],
    enemy_xy: Sequence[float],
    enemy_vel_xy: Sequence[float],
    origin_xy: Sequence[float],
) -> float:
    if not DEF_NEAR_ENEMY_SLOW_ENABLED:
        return 1.0

    rel = np.asarray(def_xy, dtype=np.float32) - np.asarray(enemy_xy, dtype=np.float32)
    dist = float(np.linalg.norm(rel))
    if dist >= DEF_SLOW_DIST_FAR:
        return 1.0

    to_origin = np.asarray(origin_xy, dtype=np.float32) - np.asarray(enemy_xy, dtype=np.float32)
    goal_norm = float(np.linalg.norm(to_origin))
    if goal_norm < 1e-9:
        return 1.0
    u_goal = to_origin / (goal_norm + 1e-12)
    u_lat = np.array([-u_goal[1], u_goal[0]], dtype=np.float32)

    vel_xy = np.asarray(enemy_vel_xy, dtype=np.float32)
    vel_norm = float(np.linalg.norm(vel_xy))
    if vel_norm > 1e-6:
        approach_cos = float(np.dot(vel_xy / (vel_norm + 1e-12), u_goal))
        if approach_cos < DEF_SLOW_APPROACH_COS:
            return 1.0

    proj = float(np.dot(rel, u_goal))
    lat = abs(float(np.dot(rel, u_lat)))
    if proj <= DEF_SLOW_MIN_PROJ or lat >= DEF_SLOW_LATERAL:
        return 1.0

    if dist <= DEF_SLOW_DIST_NEAR:
        return float(DEF_SLOW_SCALE_CLOSE)
    if dist <= DEF_SLOW_DIST_MID:
        t = (dist - DEF_SLOW_DIST_NEAR) / max(1e-6, DEF_SLOW_DIST_MID - DEF_SLOW_DIST_NEAR)
        return float(DEF_SLOW_SCALE_CLOSE * (1.0 - t) + DEF_SLOW_SCALE_NEAR * t)
    t = (dist - DEF_SLOW_DIST_MID) / max(1e-6, DEF_SLOW_DIST_FAR - DEF_SLOW_DIST_MID)
    return float(DEF_SLOW_SCALE_NEAR * (1.0 - t) + DEF_SLOW_SCALE_FAR * t)


@dataclass
class VehicleState:
    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self) -> None:
        self.position = _as_np3(self.position, name="position")
        self.velocity = _as_np3(self.velocity, name="velocity")


@dataclass
class InferenceSnapshot:
    defenders: List[VehicleState]
    enemy: VehicleState
    step_count: int = 0
    active_defender_indices: Tuple[int, ...] = (0, 1, 2, 3)

    def __post_init__(self) -> None:
        if len(self.defenders) != 4:
            raise ValueError(f"当前模型固定支持 4 架 defender，实际得到 {len(self.defenders)}")
        indices = tuple(int(idx) for idx in self.active_defender_indices)
        if not indices:
            raise ValueError("active_defender_indices 至少需要包含一架 defender")
        if len(set(indices)) != len(indices):
            raise ValueError(f"active_defender_indices 不能重复: {indices}")
        for idx in indices:
            if idx < 0 or idx >= len(self.defenders):
                raise ValueError(f"active_defender_indices 包含非法索引 {idx}")
        self.active_defender_indices = indices


@dataclass
class RoleAssignment:
    roles: np.ndarray
    sides: np.ndarray
    slot_targets: np.ndarray
    slot_names: List[str]
    meta: Dict[str, Any]
    refresh_step: int


@dataclass
class DefenderCommand:
    name: str
    action_xy_norm: np.ndarray
    velocity_xyz: np.ndarray
    speed_limit: float
    role_name: str
    slot_target_xy: np.ndarray


@dataclass
class InferenceResult:
    observations: List[np.ndarray]
    normalized_actions: List[np.ndarray]
    commands: List[DefenderCommand]
    role_assignment: RoleAssignment


@dataclass
class InferenceConfig:
    config_path: str = DEFAULT_CONFIG_PATH
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH
    device: str = str(_cfg(_RUNTIME_CONFIG, "inference", "device", "auto"))
    defender_names: Tuple[str, ...] = tuple(_cfg(_RUNTIME_CONFIG, "inference", "defender_names", ("Drone1", "Drone2", "Drone3", "Drone4")))
    origin: Tuple[float, float, float] = tuple(_cfg(_RUNTIME_CONFIG, "inference", "origin", (0.0, 0.0, 1.0)))
    obs_dim: int = int(_cfg(_RUNTIME_CONFIG, "inference", "obs_dim", 25))
    act_dim: int = int(_cfg(_RUNTIME_CONFIG, "inference", "act_dim", 2))
    shared_actor: bool = bool(_cfg(_RUNTIME_CONFIG, "inference", "shared_actor", True))
    pos_scale: float = float(_cfg(_RUNTIME_CONFIG, "inference", "pos_scale", 90.0))
    vel_scale: float = float(_cfg(_RUNTIME_CONFIG, "inference", "vel_scale", 10.0))
    defender_vmax: float = DEF_VMAX
    desired_altitude_z: Optional[float] = _cfg(_RUNTIME_CONFIG, "inference", "desired_altitude_z", None)
    z_kp: float = float(_cfg(_RUNTIME_CONFIG, "inference", "z_kp", 1.8))
    z_kd: float = float(_cfg(_RUNTIME_CONFIG, "inference", "z_kd", 0.8))
    z_vz_limit: float = float(_cfg(_RUNTIME_CONFIG, "inference", "z_vz_limit", 3.5))
    z_soft_margin: float = float(_cfg(_RUNTIME_CONFIG, "inference", "z_soft_margin", 1.0))
    z_hard_push: float = float(_cfg(_RUNTIME_CONFIG, "inference", "z_hard_push", 2.6))
    apply_role_guidance: bool = bool(_cfg(_RUNTIME_CONFIG, "inference", "apply_role_guidance", True))
    apply_safety_guard: bool = bool(_cfg(_RUNTIME_CONFIG, "inference", "apply_safety_guard", True))

    def __post_init__(self) -> None:
        self.checkpoint_path = _resolve_path(str(self.checkpoint_path), MODULE_DIR)
        self.defender_names = tuple(str(name) for name in self.defender_names)
        self.origin = tuple(float(v) for v in self.origin)
        if len(self.defender_names) != 4:
            raise ValueError(f"defender_names 必须配置 4 个名称，实际得到 {len(self.defender_names)}")
        if len(self.origin) != 3:
            raise ValueError(f"origin 必须是长度为 3 的数组，实际得到 {len(self.origin)}")
        if self.desired_altitude_z is not None:
            self.desired_altitude_z = float(self.desired_altitude_z)

    @classmethod
    def from_yaml(cls, config_path: str = DEFAULT_CONFIG_PATH) -> "InferenceConfig":
        path = Path(config_path)
        data = _load_yaml_config(str(path))
        base_dir = path.resolve().parent
        _apply_runtime_config(data, base_dir)
        inference = _section(data, "inference")

        return cls(
            config_path=str(path),
            checkpoint_path=_resolve_path(str(inference.get("checkpoint_path", "policy_checkpoint.pt")), base_dir),
            device=str(inference.get("device", "auto")),
            defender_names=tuple(inference.get("defender_names", ("Drone1", "Drone2", "Drone3", "Drone4"))),
            origin=tuple(inference.get("origin", (0.0, 0.0, 1.0))),
            obs_dim=int(inference.get("obs_dim", 25)),
            act_dim=int(inference.get("act_dim", 2)),
            shared_actor=bool(inference.get("shared_actor", True)),
            pos_scale=float(inference.get("pos_scale", 90.0)),
            vel_scale=float(inference.get("vel_scale", 10.0)),
            defender_vmax=DEF_VMAX,
            desired_altitude_z=inference.get("desired_altitude_z", None),
            z_kp=float(inference.get("z_kp", 1.8)),
            z_kd=float(inference.get("z_kd", 0.8)),
            z_vz_limit=float(inference.get("z_vz_limit", 3.5)),
            z_soft_margin=float(inference.get("z_soft_margin", 1.0)),
            z_hard_push=float(inference.get("z_hard_push", 2.6)),
            apply_role_guidance=bool(inference.get("apply_role_guidance", True)),
            apply_safety_guard=bool(inference.get("apply_safety_guard", True)),
        )

    def resolved_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    @property
    def origin_array(self) -> np.ndarray:
        return np.asarray(self.origin, dtype=np.float32)

    @property
    def origin_xy(self) -> np.ndarray:
        return self.origin_array[:2]

    @property
    def target_altitude_z(self) -> float:
        if self.desired_altitude_z is not None:
            return float(self.desired_altitude_z)
        return float(self.origin[2])


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, extra_dim: int = 0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + extra_dim, 160),
            nn.ReLU(),
            nn.Linear(160, 160),
            nn.ReLU(),
            nn.Linear(160, act_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RolePlanner:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.current_roles = np.array(
            [ROLE_INTERCEPTOR, ROLE_INTERCEPTOR, ROLE_INTERCEPTOR, ROLE_BLOCKER],
            dtype=np.int64,
        )
        self.current_sides = np.zeros((4,), dtype=np.int64)
        self.current_slot_targets = np.zeros((4, 2), dtype=np.float32)
        self.current_slot_names = ["init"] * 4
        self.current_role_meta: Dict[str, Any] = {}
        self.current_role_refresh_step = -10**9
        self.enemy_orbit_latch = 0

    @staticmethod
    def _norm_xy(v: Sequence[float], fallback: Optional[Sequence[float]] = None) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 1e-9:
            return arr / (norm + 1e-12)
        if fallback is None:
            return np.array([1.0, 0.0], dtype=np.float32)
        fb = np.asarray(fallback, dtype=np.float32)
        fb_norm = float(np.linalg.norm(fb))
        if fb_norm > 1e-9:
            return fb / (fb_norm + 1e-12)
        return np.array([1.0, 0.0], dtype=np.float32)

    @staticmethod
    def _perp_xy(u: Sequence[float]) -> np.ndarray:
        arr = np.asarray(u, dtype=np.float32)
        return np.array([-arr[1], arr[0]], dtype=np.float32)

    def _role_space_scale(self, enemy_xy: np.ndarray) -> float:
        eo = float(np.linalg.norm(self.config.origin_xy - enemy_xy))
        scale = 1.0
        if eo <= ROLE_KILL_TRIGGER:
            scale = 0.84
        if int(self.enemy_orbit_latch) > 0:
            scale = min(scale, ROLE_ORBIT_SCALE)
        return float(scale)

    def _defender_kill_mode_active(self, enemy_pos: np.ndarray, step_count: int) -> bool:
        enemy_xy = enemy_pos[:2].astype(np.float32)
        eo = float(np.linalg.norm(self.config.origin_xy - enemy_xy))
        tfrac = float(step_count) / max(1.0, 220.0)
        return bool((eo <= ROLE_KILL_TRIGGER) or ((tfrac >= ROLE_KILL_LATE_FRAC) and (eo <= ROLE_KILL_LATE_DIST)))

    def _build_role_slots(
        self,
        enemy_pos: np.ndarray,
        enemy_vel: np.ndarray,
        step_count: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
        enemy_xy = enemy_pos[:2].astype(np.float32)
        origin_xy = self.config.origin_xy.astype(np.float32)
        to_origin = origin_xy - enemy_xy
        eo = float(np.linalg.norm(to_origin))
        u_goal = self._norm_xy(to_origin, np.array([1.0, 0.0], dtype=np.float32))
        u_lat = self._perp_xy(u_goal)
        v_dir = self._norm_xy(enemy_vel[:2], u_goal)
        goal_align = float(np.dot(v_dir, u_goal))
        if goal_align < 0.10:
            v_dir = self._norm_xy(0.20 * v_dir + 0.80 * u_goal, u_goal)
        elif eo <= ROLE_BREAK_ENEMY_DIST:
            v_dir = self._norm_xy(0.55 * v_dir + 0.45 * u_goal, u_goal)

        scale = self._role_space_scale(enemy_xy)
        kill_mode = self._defender_kill_mode_active(enemy_pos, step_count)

        guard_forward = min(ROLE_GUARD_FORWARD_OFFSET * scale, max(1.0, eo - 0.5))
        guard_slot = origin_xy - u_goal * guard_forward

        if kill_mode:
            atk_lat = float(np.clip(0.16 * eo + 1.4, ROLE_ATTACK_LAT_MIN, ROLE_ATTACK_LAT_MAX - 0.2))
            atk_forward = min(2.35, 1.05 + 0.055 * eo)
            catch_forward = min(1.25, 0.42 + 0.030 * eo)
            atk_left = enemy_xy + u_goal * atk_forward + u_lat * atk_lat
            atk_right = enemy_xy + u_goal * atk_forward - u_lat * atk_lat
            support = enemy_xy + u_goal * catch_forward
        else:
            atk_lat = float(np.clip(0.12 * eo + 1.5, ROLE_ATTACK_LAT_MIN, ROLE_ATTACK_LAT_MAX - 0.6)) * scale
            atk_forward = min(0.55 + 0.025 * eo, 1.5) * scale
            sup_forward = min(max(ROLE_SUPPORT_FORWARD * scale, atk_forward + 1.0), max(1.6, eo - 1.0))
            atk_left = enemy_xy + u_goal * atk_forward + u_lat * atk_lat
            atk_right = enemy_xy + u_goal * atk_forward - u_lat * atk_lat
            support = enemy_xy + u_goal * sup_forward

        slots = np.stack([guard_slot, atk_left, support, atk_right], axis=0).astype(np.float32)
        slot_roles = np.array([ROLE_BLOCKER, ROLE_INTERCEPTOR, ROLE_INTERCEPTOR, ROLE_INTERCEPTOR], dtype=np.int64)
        slot_sides = np.array([0, 1, 0, -1], dtype=np.int64)
        slot_names = ["gate", "atk_L", "sup", "atk_R"]
        meta = {
            "u_goal": u_goal,
            "u_lat": u_lat,
            "v_dir": v_dir,
            "gate_center": guard_slot,
            "int_anchor": enemy_xy + u_goal * (1.45 if kill_mode else 1.0),
            "scale": float(scale),
            "enemy_xy": enemy_xy,
            "origin_xy": origin_xy,
            "enemy_origin_dist": float(eo),
            "kill_mode": bool(kill_mode),
        }
        return slots, slot_roles, slot_sides, slot_names, meta

    def _solve_role_slot_assignment(
        self,
        def_xy: np.ndarray,
        slots: np.ndarray,
        slot_names: Sequence[str],
        enemy_xy: np.ndarray,
    ) -> Tuple[int, ...]:
        best_cost = None
        best_perm: Optional[Tuple[int, ...]] = None
        origin_xy = self.config.origin_xy.astype(np.float32)
        prev_names = self.current_slot_names

        for perm in itertools.permutations(range(len(slots)), len(def_xy)):
            total = 0.0
            for i, slot_idx in enumerate(perm):
                name = str(slot_names[slot_idx])
                cost = float(np.linalg.norm(def_xy[i] - slots[slot_idx]))
                if name == "gate":
                    cost += 0.10 * float(np.linalg.norm(def_xy[i] - origin_xy))
                else:
                    cost += 0.03 * max(0.0, float(np.linalg.norm(def_xy[i] - enemy_xy)) - 9.0)
                if i < len(prev_names) and str(prev_names[i]) == name:
                    cost -= 0.30 if name == "gate" else 0.18
                total += cost
            if best_cost is None or total < best_cost:
                best_cost = total
                best_perm = perm

        if best_perm is None:
            raise RuntimeError("角色分配失败")
        return best_perm

    def _should_break_role_hold(
        self,
        def_pos: np.ndarray,
        enemy_pos: np.ndarray,
        step_count: int,
        active_indices: Sequence[int],
    ) -> bool:
        if self.current_role_refresh_step < 0:
            return False
        hold_age = int(step_count - self.current_role_refresh_step)
        if hold_age < ROLE_REPLAN_MIN_STEPS:
            return False

        enemy_xy = enemy_pos[:2].astype(np.float32)
        eo = float(np.linalg.norm(self.config.origin_xy - enemy_xy))
        kill_mode = self._defender_kill_mode_active(enemy_pos, step_count)
        if kill_mode and hold_age < ROLE_KILL_HOLD_STEPS:
            return False
        if eo > ROLE_BREAK_ENEMY_DIST:
            return False

        def_xy = def_pos[:, :2].astype(np.float32)
        u_goal = self._norm_xy(self.config.origin_xy - enemy_xy, np.array([1.0, 0.0], dtype=np.float32))
        u_lat = self._perp_xy(u_goal)
        gate_bad = 0
        attack_bad = 0

        for idx in active_indices:
            slot_name = str(self.current_slot_names[idx]) if idx < len(self.current_slot_names) else ""
            slot_err = float(np.linalg.norm(def_xy[idx] - self.current_slot_targets[idx]))
            rel = (def_xy[idx] - enemy_xy).astype(np.float32)
            proj = float(np.dot(rel, u_goal) / 18.0)
            lat = abs(float(np.dot(rel, u_lat) / 12.0))
            if slot_name == "gate":
                if slot_err > ROLE_GATE_SLOT_SCALE * ROLE_BREAK_SLOT_ERR_SCALE:
                    gate_bad += 1
            else:
                if slot_err > ROLE_INTERCEPT_SLOT_SCALE:
                    attack_bad += 1
                if proj > DEFENDER_OVERSHOOT_PROJ and lat > DEFENDER_OVERSHOOT_LAT:
                    attack_bad += 1
        return (gate_bad >= 1) or (attack_bad >= 1)

    def refresh(self, snapshot: InferenceSnapshot, force: bool = False) -> RoleAssignment:
        step_count = int(snapshot.step_count)
        active_indices = tuple(snapshot.active_defender_indices)
        def_pos_all = np.stack([d.position for d in snapshot.defenders], axis=0).astype(np.float32)
        def_pos = def_pos_all[list(active_indices)]
        enemy_pos = snapshot.enemy.position.astype(np.float32)
        enemy_vel = snapshot.enemy.velocity.astype(np.float32)

        need_refresh = force or (self.current_role_refresh_step < 0) or (self.current_slot_targets is None)
        kill_mode_next = self._defender_kill_mode_active(enemy_pos, step_count)

        if not need_refresh:
            hold_age = int(step_count - self.current_role_refresh_step)
            kill_mode_now = bool(self.current_role_meta.get("kill_mode", False))
            if kill_mode_next != kill_mode_now:
                need_refresh = True
            elif hold_age >= int(ROLE_HOLD_STEPS if not kill_mode_next else ROLE_KILL_HOLD_STEPS):
                need_refresh = True
            elif self._should_break_role_hold(def_pos_all, enemy_pos, step_count, active_indices):
                need_refresh = True

        if need_refresh:
            slots, slot_roles, slot_sides, slot_names, meta = self._build_role_slots(enemy_pos, enemy_vel, step_count)
            meta["kill_mode"] = bool(self._defender_kill_mode_active(enemy_pos, step_count))
            perm = self._solve_role_slot_assignment(def_pos[:, :2], slots, slot_names, meta["enemy_xy"])

            roles = np.zeros((4,), dtype=np.int64)
            sides = np.zeros((4,), dtype=np.int64)
            targets = def_pos_all[:, :2].astype(np.float32).copy()
            names: List[str] = ["inactive"] * 4
            for local_idx, slot_idx in enumerate(perm):
                idx = int(active_indices[local_idx])
                roles[idx] = int(slot_roles[slot_idx])
                sides[idx] = int(slot_sides[slot_idx])
                targets[idx] = np.asarray(slots[slot_idx], dtype=np.float32)
                names[idx] = str(slot_names[slot_idx])

            self.current_roles = roles
            self.current_sides = sides
            self.current_slot_targets = targets
            self.current_slot_names = names
            self.current_role_meta = meta
            self.current_role_refresh_step = step_count

        return RoleAssignment(
            roles=self.current_roles.copy(),
            sides=self.current_sides.copy(),
            slot_targets=self.current_slot_targets.copy(),
            slot_names=list(self.current_slot_names),
            meta=dict(self.current_role_meta),
            refresh_step=int(self.current_role_refresh_step),
        )


class ObservationBuilder:
    def __init__(self, config: InferenceConfig):
        self.config = config

    def build(self, snapshot: InferenceSnapshot, role_assignment: RoleAssignment) -> List[np.ndarray]:
        def_p = np.stack([d.position for d in snapshot.defenders], axis=0).astype(np.float32)
        def_v = np.stack([d.velocity for d in snapshot.defenders], axis=0).astype(np.float32)
        e_p = snapshot.enemy.position.astype(np.float32)
        e_to_o = self.config.origin_array - e_p
        active_set = set(int(idx) for idx in snapshot.active_defender_indices)

        origin_xy = self.config.origin_xy.astype(np.float32)
        enemy_xy = e_p[:2].astype(np.float32)
        u_goal = np.asarray(role_assignment.meta.get("u_goal"), dtype=np.float32)
        u_lat = np.asarray(role_assignment.meta.get("u_lat"), dtype=np.float32)
        hold_left = max(0, int(ROLE_HOLD_STEPS) - int(snapshot.step_count - role_assignment.refresh_step))
        hold_norm = float(hold_left) / float(max(1, ROLE_HOLD_STEPS))
        enemy_to_origin_norm = float(np.linalg.norm(origin_xy - enemy_xy) / max(1.0, 55.0))

        obs_list: List[np.ndarray] = []
        for i in range(4):
            rel_e = (e_p - def_p[i]) / float(self.config.pos_scale)
            eto = e_to_o / float(self.config.pos_scale)
            sv = def_v[i] / float(self.config.vel_scale)

            rels: List[float] = []
            for j in range(4):
                if j == i:
                    continue
                if j not in active_set:
                    rels.extend([0.0, 0.0])
                    continue
                dij = (def_p[j] - def_p[i]) / float(self.config.pos_scale)
                rels.extend([float(dij[0]), float(dij[1])])

            role_int = 1.0 if int(role_assignment.roles[i]) == ROLE_INTERCEPTOR else 0.0
            role_blk = 1.0 - role_int
            side_left = 1.0 if int(role_assignment.sides[i]) > 0 else 0.0
            side_right = 1.0 if int(role_assignment.sides[i]) < 0 else 0.0
            slot_rel = (role_assignment.slot_targets[i] - def_p[i][:2]) / float(ROLE_SLOT_OBS_SCALE)
            rel_enemy_xy = (def_p[i][:2] - enemy_xy).astype(np.float32)
            along_goal = float(np.dot(rel_enemy_xy, u_goal) / 18.0)
            lateral_goal = float(np.dot(rel_enemy_xy, u_lat) / 12.0)

            obs = np.array(
                [
                    rel_e[0], rel_e[1], rel_e[2],
                    eto[0], eto[1], eto[2],
                    sv[0], sv[1], sv[2],
                    *rels,
                    role_int, role_blk, side_left, side_right,
                    slot_rel[0], slot_rel[1],
                    hold_norm, along_goal, lateral_goal, enemy_to_origin_norm,
                ],
                dtype=np.float32,
            )
            if obs.shape != (self.config.obs_dim,):
                raise RuntimeError(f"观测维度错误，期望 {self.config.obs_dim}，实际 {obs.shape}")
            obs_list.append(obs)
        return obs_list


class CommandPostprocessor:
    def __init__(self, config: InferenceConfig):
        self.config = config

    @staticmethod
    def _norm_xy(v: Sequence[float], fallback: Optional[Sequence[float]] = None) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 1e-9:
            return arr / (norm + 1e-12)
        if fallback is None:
            return np.array([1.0, 0.0], dtype=np.float32)
        fb = np.asarray(fallback, dtype=np.float32)
        fb_norm = float(np.linalg.norm(fb))
        if fb_norm > 1e-9:
            return fb / (fb_norm + 1e-12)
        return np.array([1.0, 0.0], dtype=np.float32)

    def _apply_safety(
        self,
        index: int,
        vx: float,
        vy: float,
        def_pos: np.ndarray,
        active_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[float, float]:
        pi = def_pos[index]
        active_iter = range(len(def_pos)) if active_indices is None else active_indices
        for j in active_iter:
            if j == index:
                continue
            pj = def_pos[j]
            dxy = pj[:2] - pi[:2]
            dist = float(np.linalg.norm(dxy)) + 1e-9
            if dist < REPULSE_RANGE:
                away = -dxy / dist
                strength = REPULSE_GAIN * (1.0 / dist - 1.0 / REPULSE_RANGE)
                vx += float(away[0] * strength)
                vy += float(away[1] * strength)

        x, y = float(pi[0]), float(pi[1])
        if x > X_MAX - BOUNDARY_MARGIN:
            vx = min(vx, 0.0)
            vx -= BOUNDARY_GAIN
        if x < X_MIN + BOUNDARY_MARGIN:
            vx = max(vx, 0.0)
            vx += BOUNDARY_GAIN
        if y > Y_MAX - BOUNDARY_MARGIN:
            vy = min(vy, 0.0)
            vy -= BOUNDARY_GAIN
        if y < Y_MIN + BOUNDARY_MARGIN:
            vy = max(vy, 0.0)
            vy += BOUNDARY_GAIN
        return clip_vec2(vx, vy, self.config.defender_vmax)

    def _hold_alt_vz(self, current_z: float, current_vz: float, target_z: float) -> float:
        err_z = float(target_z - current_z)
        vz_cmd = float(self.config.z_kp * err_z - self.config.z_kd * current_vz)
        if current_z > (Z_MAX - self.config.z_soft_margin):
            vz_cmd = min(vz_cmd, -self.config.z_hard_push - 0.8 * max(0.0, current_vz))
        if current_z < (Z_MIN + self.config.z_soft_margin):
            vz_cmd = max(vz_cmd, self.config.z_hard_push + 0.8 * max(0.0, -current_vz))
        return float(np.clip(vz_cmd, -self.config.z_vz_limit, self.config.z_vz_limit))

    def _defender_role_guidance(
        self,
        idx: int,
        p: np.ndarray,
        v: np.ndarray,
        e_p: np.ndarray,
        role_assignment: RoleAssignment,
    ) -> Tuple[np.ndarray, float]:
        enemy_xy = e_p[:2].astype(np.float32)
        origin_xy = self.config.origin_xy.astype(np.float32)
        slot = np.asarray(role_assignment.slot_targets[idx], dtype=np.float32)
        slot_name = str(role_assignment.slot_names[idx]) if idx < len(role_assignment.slot_names) else ""
        meta = role_assignment.meta
        u_goal = np.asarray(meta.get("u_goal"), dtype=np.float32)
        u_lat = np.asarray(meta.get("u_lat"), dtype=np.float32)
        kill_mode = bool(meta.get("kill_mode", False))
        rel_enemy = enemy_xy - p[:2]
        guide_to_origin = origin_xy - enemy_xy
        blend = float(DEFENDER_OVERRIDE_BLEND_KILL if kill_mode else DEFENDER_OVERRIDE_BLEND)
        d_enemy = float(np.linalg.norm(rel_enemy))
        eo = float(np.linalg.norm(guide_to_origin))

        if slot_name == "gate":
            desired = 1.10 * (slot - p[:2]) + 0.55 * rel_enemy + 0.18 * guide_to_origin - 0.18 * v[:2]
            if kill_mode:
                desired = 1.10 * (slot - p[:2]) + 0.55 * rel_enemy + 0.78 * guide_to_origin - 0.20 * v[:2]
            blend = max(blend, DEFENDER_OVERRIDE_BLEND_GATE)
        elif slot_name == "sup":
            desired = 0.35 * (slot - p[:2]) + 1.22 * rel_enemy + 0.30 * guide_to_origin - 0.18 * v[:2]
            if kill_mode:
                desired = 1.35 * (slot - p[:2]) + 0.62 * rel_enemy + 0.90 * guide_to_origin - 0.24 * v[:2]
        else:
            desired = 0.25 * (slot - p[:2]) + 1.35 * rel_enemy + 0.28 * guide_to_origin - 0.20 * v[:2]
            if kill_mode:
                desired = 1.60 * (slot - p[:2]) + 0.52 * rel_enemy + 0.95 * guide_to_origin - 0.26 * v[:2]

        if slot_name != "gate":
            rel = (p[:2] - enemy_xy).astype(np.float32)
            proj = float(np.dot(rel, u_goal) / 18.0)
            lat = abs(float(np.dot(rel, u_lat) / 12.0))
            if proj > DEFENDER_OVERSHOOT_PROJ and lat > DEFENDER_OVERSHOOT_LAT:
                desired = 0.75 * rel_enemy + 1.05 * guide_to_origin - 1.05 * v[:2]
                blend = max(blend, DEFENDER_OVERSHOOT_BLEND)

        if (d_enemy <= DEFENDER_FINISH_DIST) and (eo <= DEFENDER_FINISH_EO):
            desired = 1.80 * rel_enemy + 0.22 * guide_to_origin - 0.10 * v[:2]
            if slot_name == "gate" and (not kill_mode):
                desired = 1.20 * (slot - p[:2]) + 1.25 * rel_enemy + 0.18 * guide_to_origin - 0.10 * v[:2]
            blend = max(blend, DEFENDER_FINISH_BLEND)

        return desired.astype(np.float32), float(np.clip(blend, 0.0, 0.985))

    def build_commands(
        self,
        snapshot: InferenceSnapshot,
        raw_actions: List[np.ndarray],
        role_assignment: RoleAssignment,
    ) -> List[DefenderCommand]:
        def_pos = np.stack([d.position for d in snapshot.defenders], axis=0).astype(np.float32)
        enemy_pos = snapshot.enemy.position.astype(np.float32)
        enemy_vel = snapshot.enemy.velocity.astype(np.float32)
        commands: List[DefenderCommand] = []

        for i, (name, defender) in enumerate(zip(self.config.defender_names, snapshot.defenders)):
            a = np.clip(np.asarray(raw_actions[i], dtype=np.float32), -1.0, 1.0)
            p = defender.position.astype(np.float32)
            v = defender.velocity.astype(np.float32)
            vmax_local = float(
                self.config.defender_vmax
                * defender_speed_scale_by_enemy_dist(p[:2], enemy_pos[:2], enemy_vel[:2], self.config.origin_xy)
            )
            if int(role_assignment.roles[i]) == ROLE_INTERCEPTOR:
                vmax_local = max(vmax_local, self.config.defender_vmax * DEF_INTERCEPTOR_SPEED_FLOOR)
            else:
                vmax_local = min(self.config.defender_vmax * DEF_BLOCKER_SPEED_SCALE, vmax_local)

            if self.config.apply_role_guidance:
                guide_xy, guide_blend = self._defender_role_guidance(i, p, v, enemy_pos, role_assignment)
                guide_norm = float(np.linalg.norm(guide_xy))
                if guide_norm > 1e-6:
                    guide_u = guide_xy / guide_norm
                    raw_norm = float(np.linalg.norm(a))
                    raw_u = a / raw_norm if raw_norm > 1e-6 else guide_u
                    a_eff = self._norm_xy((1.0 - guide_blend) * raw_u + guide_blend * guide_u, guide_u)
                else:
                    a_eff = a
            else:
                a_eff = a

            vx, vy = float(a_eff[0] * vmax_local), float(a_eff[1] * vmax_local)
            if self.config.apply_safety_guard:
                vx, vy = self._apply_safety(i, vx, vy, def_pos, snapshot.active_defender_indices)
            vx, vy = clip_vec2(vx, vy, vmax_local)
            vz = self._hold_alt_vz(float(p[2]), float(v[2]), self.config.target_altitude_z)

            commands.append(
                DefenderCommand(
                    name=name,
                    action_xy_norm=np.asarray(a, dtype=np.float32),
                    velocity_xyz=np.array([vx, vy, vz], dtype=np.float32),
                    speed_limit=float(vmax_local),
                    role_name=str(role_assignment.slot_names[i]),
                    slot_target_xy=np.asarray(role_assignment.slot_targets[i], dtype=np.float32),
                )
            )
        return commands


class InferenceEngine:
    def __init__(self, config: Optional[InferenceConfig] = None, config_path: Optional[str] = None):
        self.config = config or InferenceConfig.from_yaml(config_path or DEFAULT_CONFIG_PATH)
        self.device = self.config.resolved_device()
        self.extra_dim = len(self.config.defender_names) if self.config.shared_actor else 0
        self.actor = Actor(self.config.obs_dim, self.config.act_dim, extra_dim=self.extra_dim).to(self.device)
        self.role_planner = RolePlanner(self.config)
        self.obs_builder = ObservationBuilder(self.config)
        self.postprocessor = CommandPostprocessor(self.config)
        self.checkpoint_meta: Dict[str, Any] = {}
        self._load_checkpoint(self.config.checkpoint_path)

    def _id_onehot(self, idx: int, batch_size: int = 1) -> torch.Tensor:
        vec = torch.zeros((batch_size, len(self.config.defender_names)), device=self.device)
        vec[:, idx] = 1.0
        return vec

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")

        payload = torch.load(str(ckpt_path), map_location=self.device, weights_only=True)
        if "actor" not in payload:
            raise KeyError("checkpoint 中未找到 actor 权重")
        self.actor.load_state_dict(payload["actor"])
        self.actor.eval()
        self.checkpoint_meta = {"ep": payload.get("ep"), "path": str(ckpt_path)}

    def reset(self) -> None:
        self.role_planner.reset()

    @torch.no_grad()
    def predict_from_observations(self, obs_list: Sequence[Sequence[float]]) -> List[np.ndarray]:
        if len(obs_list) != len(self.config.defender_names):
            raise ValueError(f"观测数量应为 {len(self.config.defender_names)}，实际得到 {len(obs_list)}")

        actions: List[np.ndarray] = []
        for i, obs in enumerate(obs_list):
            obs_arr = np.asarray(obs, dtype=np.float32)
            if obs_arr.shape != (self.config.obs_dim,):
                raise ValueError(f"第 {i} 个观测维度错误，期望 {(self.config.obs_dim,)}, 实际 {obs_arr.shape}")
            obs_tensor = torch.tensor(obs_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
            if self.config.shared_actor:
                actor_input = torch.cat([obs_tensor, self._id_onehot(i, 1)], dim=-1)
            else:
                actor_input = obs_tensor
            action = self.actor(actor_input).squeeze(0).cpu().numpy().astype(np.float32)
            actions.append(np.clip(action, -1.0, 1.0))
        return actions

    def predict(self, snapshot: InferenceSnapshot) -> InferenceResult:
        role_assignment = self.role_planner.refresh(snapshot, force=False)
        observations = self.obs_builder.build(snapshot, role_assignment)
        normalized_actions = self.predict_from_observations(observations)
        commands = self.postprocessor.build_commands(snapshot, normalized_actions, role_assignment)
        return InferenceResult(
            observations=observations,
            normalized_actions=normalized_actions,
            commands=commands,
            role_assignment=role_assignment,
        )
