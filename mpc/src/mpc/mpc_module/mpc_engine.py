#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .enemy_strategy import DEFAULT_CONFIG_PATH, EnemyStrategyMPC


def _as_np3(values: Sequence[float], *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError(f"{name} 必须是长度为 3 的向量，实际得到 {arr.shape}")
    return arr


@dataclass
class VehicleState:
    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self) -> None:
        self.position = _as_np3(self.position, name="position")
        self.velocity = _as_np3(self.velocity, name="velocity")


@dataclass
class MpcSnapshot:
    enemy: VehicleState
    defenders: List[VehicleState]
    step_count: int = 0
    reset: bool = False
    overtime_assault_active: bool = False

    def __post_init__(self) -> None:
        if len(self.defenders) == 0:
            raise ValueError("defenders 不能为空")


@dataclass
class MpcPlanResult:
    velocity_xyz: np.ndarray
    yaw_rad: float
    yaw_deg: float
    predicted_position: np.ndarray
    step_count: int
    debug: Dict[str, Any]
    raw: Dict[str, Any]


class MpcEngine:
    def __init__(self, config_path: str = str(DEFAULT_CONFIG_PATH)):
        self.config_path = str(config_path)
        self.policy = EnemyStrategyMPC.from_file(Path(config_path))

    def reset(self) -> None:
        self.policy.reset()

    def plan(self, snapshot: MpcSnapshot) -> MpcPlanResult:
        if snapshot.reset:
            self.reset()

        raw = self.policy.plan(
            enemy_pos=snapshot.enemy.position,
            enemy_vel=snapshot.enemy.velocity,
            defender_positions=[defender.position for defender in snapshot.defenders],
            step_count=int(snapshot.step_count),
            overtime_assault_active=bool(snapshot.overtime_assault_active),
        )
        velocity_xy = np.asarray(raw["velocity_xy"], dtype=np.float32)
        velocity_xyz = np.asarray([velocity_xy[0], velocity_xy[1], 0.0], dtype=np.float32)
        return MpcPlanResult(
            velocity_xyz=velocity_xyz,
            yaw_rad=float(raw["yaw_rad"]),
            yaw_deg=float(raw["yaw_deg"]),
            predicted_position=np.asarray(raw["predicted_position"], dtype=np.float32),
            step_count=int(raw["step_count"]),
            debug=dict(raw.get("debug", {})),
            raw=raw,
        )
