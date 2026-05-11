#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


ENEMY_ROLE = "enemy"
DEFENDER_ROLES = ("defender_0", "defender_1", "defender_2", "defender_3")
ALL_ROLES = (ENEMY_ROLE,) + DEFENDER_ROLES


@dataclass
class GameEndConfig:
    enabled: bool = True
    capture_distance_m: float = 1.0
    asset_distance_m: float = 1.0
    hold_duration_sec: float = 0.5
    asset_origin: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float32))
    x_bounds: Optional[Tuple[float, float]] = None
    y_bounds: Optional[Tuple[float, float]] = None
    z_bounds: Optional[Tuple[float, float]] = None
    out_of_bounds_enabled: bool = True

    def __post_init__(self) -> None:
        self.asset_origin = np.asarray(self.asset_origin, dtype=np.float32)
        if self.asset_origin.shape != (3,):
            raise ValueError(f"game_end.asset_origin 必须是长度为 3 的数组，实际得到 {self.asset_origin.shape}")
        if self.capture_distance_m <= 0.0:
            raise ValueError("game_end.capture_distance_m 必须大于 0")
        if self.asset_distance_m <= 0.0:
            raise ValueError("game_end.asset_distance_m 必须大于 0")
        if self.hold_duration_sec < 0.0:
            raise ValueError("game_end.hold_duration_sec 不能小于 0")


@dataclass(frozen=True)
class GameEndStatus:
    active: bool = False
    outcome: str = ""
    reason: str = ""
    trigger_roles: Tuple[str, ...] = ()
    trigger_distance_m: Optional[float] = None
    triggered_at_sec: Optional[float] = None


class GameEndMonitor:
    def __init__(self, config: GameEndConfig):
        self.config = config
        self.status = GameEndStatus()
        self._capture_start_sec: Optional[float] = None
        self._asset_start_sec: Optional[float] = None

    def reset(self) -> None:
        self.status = GameEndStatus()
        self._capture_start_sec = None
        self._asset_start_sec = None

    def update(self, positions: Mapping[str, Sequence[float]], now_sec: float) -> GameEndStatus:
        if self.status.active or not self.config.enabled:
            return self.status

        normalized = self._normalize_positions(positions)
        out_of_bounds = self._first_out_of_bounds(normalized)
        if out_of_bounds is not None:
            role, axis, value = out_of_bounds
            return self._activate(
                outcome="experiment_failed",
                reason=f"{role} out of world bounds on {axis}: {value:.3f}",
                trigger_roles=(role,),
                trigger_distance_m=None,
                now_sec=now_sec,
            )

        capture = self._capture_candidate(normalized)
        if capture is not None:
            role, distance = capture
            self._capture_start_sec = now_sec if self._capture_start_sec is None else self._capture_start_sec
            if now_sec - self._capture_start_sec >= self.config.hold_duration_sec:
                return self._activate(
                    outcome="defender_win",
                    reason=f"{role} captured enemy for {self.config.hold_duration_sec:.3f}s",
                    trigger_roles=(role, ENEMY_ROLE),
                    trigger_distance_m=distance,
                    now_sec=now_sec,
                )
        else:
            self._capture_start_sec = None

        asset_distance = self._asset_distance(normalized)
        if asset_distance is not None and asset_distance < self.config.asset_distance_m:
            self._asset_start_sec = now_sec if self._asset_start_sec is None else self._asset_start_sec
            if now_sec - self._asset_start_sec >= self.config.hold_duration_sec:
                return self._activate(
                    outcome="enemy_win",
                    reason=f"enemy reached critical asset for {self.config.hold_duration_sec:.3f}s",
                    trigger_roles=(ENEMY_ROLE,),
                    trigger_distance_m=asset_distance,
                    now_sec=now_sec,
                )
        else:
            self._asset_start_sec = None

        return self.status

    def _normalize_positions(self, positions: Mapping[str, Sequence[float]]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for role, position in positions.items():
            arr = np.asarray(position, dtype=np.float32)
            if arr.shape == (3,):
                out[str(role)] = arr
        return out

    def _first_out_of_bounds(self, positions: Mapping[str, np.ndarray]) -> Optional[Tuple[str, str, float]]:
        if not self.config.out_of_bounds_enabled:
            return None
        for role in ALL_ROLES:
            position = positions.get(role)
            if position is None:
                continue
            checks = (
                ("x", self.config.x_bounds, float(position[0])),
                ("y", self.config.y_bounds, float(position[1])),
                ("z", self.config.z_bounds, float(position[2])),
            )
            for axis, bounds, value in checks:
                if bounds is None:
                    continue
                lower, upper = bounds
                if value < lower or value > upper:
                    return role, axis, value
        return None

    def _capture_candidate(self, positions: Mapping[str, np.ndarray]) -> Optional[Tuple[str, float]]:
        enemy = positions.get(ENEMY_ROLE)
        if enemy is None:
            return None
        best: Optional[Tuple[str, float]] = None
        for role in DEFENDER_ROLES:
            defender = positions.get(role)
            if defender is None:
                continue
            distance = float(np.linalg.norm(defender - enemy))
            if distance < self.config.capture_distance_m and (best is None or distance < best[1]):
                best = (role, distance)
        return best

    def _asset_distance(self, positions: Mapping[str, np.ndarray]) -> Optional[float]:
        enemy = positions.get(ENEMY_ROLE)
        if enemy is None:
            return None
        return float(np.linalg.norm(enemy[:2] - self.config.asset_origin[:2]))

    def _activate(
        self,
        *,
        outcome: str,
        reason: str,
        trigger_roles: Tuple[str, ...],
        trigger_distance_m: Optional[float],
        now_sec: float,
    ) -> GameEndStatus:
        self.status = GameEndStatus(
            active=True,
            outcome=outcome,
            reason=reason,
            trigger_roles=trigger_roles,
            trigger_distance_m=trigger_distance_m,
            triggered_at_sec=float(now_sec),
        )
        return self.status
