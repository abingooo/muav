import math
import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name("mpc_config.yaml")


def lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def vec2_to_yaw(v: Iterable[float]) -> float:
    arr = np.asarray(v, dtype=float)
    return float(np.arctan2(float(arr[1]), float(arr[0])))


def yaw_to_unit(yaw: float) -> np.ndarray:
    return np.array([math.cos(float(yaw)), math.sin(float(yaw))], dtype=float)


def _as_xy_array(points: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("defender_positions must have shape (n, 2) or (n, 3)")
    return arr[:, :2].astype(float)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"MPC config 顶层必须是 YAML mapping: {path}")
    return data


def _build_policy_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    env = dict(raw.get("environment", {}))
    enemy = dict(raw.get("enemy", {}))
    presets = dict(raw.get("presets", {}))
    preset = str(enemy.get("difficulty_preset", "medium")).strip().lower()
    base = dict(presets.get(preset, presets.get("medium", {})))
    overrides = dict(enemy.get("custom_overrides", {}))

    for key, value in overrides.items():
        if value is not None:
            base[key] = value

    base["preset_name"] = preset if preset in presets else "medium"
    base["candidate_count"] = int(max(5, int(base.get("candidate_count", 17))))
    base["horizon"] = int(max(1, int(base.get("horizon", 2))))
    base["reaction_steps"] = int(max(0, int(base.get("reaction_steps", 0))))
    base["dash_reaction_steps"] = int(max(0, int(base.get("dash_reaction_steps", 0))))

    base.setdefault("time_commit_start_frac", 0.50)
    base.setdefault("time_hard_commit_start_frac", 0.80)
    base.setdefault("time_goal_w_mult_end", 2.10)
    base.setdefault("time_sep_w_mult_end", 0.18)
    base.setdefault("time_surround_w_mult_end", 0.24)
    base.setdefault("time_front_pressure_w_mult_end", 0.18)
    base.setdefault("time_turn_w_mult_end", 0.30)
    base.setdefault("time_blocked_goal_w_mult_end", 2.10)
    base.setdefault("time_blocked_sep_w_mult_end", 0.26)
    base.setdefault("time_blocked_surround_w_mult_end", 0.32)
    base.setdefault("time_speed_scale_end", 1.10)
    base.setdefault("time_final_second_speed_scale", 1.35)
    base.setdefault("time_eps_random_end", 0.0)
    base.setdefault("time_jitter_deg_end", 0.0)
    base.setdefault("time_near_goal_speed_scale_end", 0.96)
    base.setdefault("time_terminal_cap_gain_end", 0.78)
    base.setdefault("time_force_goal_window_sec", 1.0)
    base.setdefault("time_arrival_margin", 0.92)
    base.setdefault("time_hard_commit_goal_half_deg", 10.0)
    base.setdefault("time_hard_commit_candidate_count", 5)
    base.setdefault("time_hard_commit_goal_w_mult", 3.20)
    base.setdefault("time_hard_commit_sep_w_mult", 0.10)
    base.setdefault("time_hard_commit_surround_w_mult", 0.10)
    base.setdefault("time_hard_commit_front_pressure_w_mult", 0.08)
    base.setdefault("time_hard_commit_turn_w_mult", 0.05)
    base.setdefault("time_hard_commit_speed_scale", 1.18)
    base.setdefault("time_hard_commit_heading_mix", 0.95)
    base.setdefault("overtime_speed_scale", 1.28)
    base.setdefault("enemy_z_band_half", env.get("enemy_z_band_half", 3.2))
    base.setdefault("enemy_z_recover_margin", env.get("enemy_z_recover_margin", 0.45))
    base.setdefault("enemy_z_global_margin", env.get("enemy_z_global_margin", 0.8))
    base.setdefault("enemy_vz_abs_limit", 4.8)
    base.setdefault("top_z_guard_margin", 1.2)
    base.setdefault("top_z_safe_margin", 0.55)
    base.setdefault("bottom_z_guard_margin", 1.2)
    base.setdefault("bottom_z_safe_margin", 0.55)
    return base


class EnemyStrategyMPC:
    """Standalone enemy mpc_lite policy.

    Input:
        enemy_pos: [x, y] or [x, y, z]
        enemy_vel: [vx, vy] or [vx, vy, vz]
        defender_positions: [[x, y, z], ...]

    Output from plan():
        dict with velocity_xy, yaw_rad, predicted_position and mode flags.
    """

    def __init__(self, config: Dict[str, Any]):
        self.raw_config = config
        env = dict(config.get("environment", {}))
        enemy = dict(config.get("enemy", {}))
        self.cfg = _build_policy_config(config)

        self.dt = float(env.get("dt", 0.2))
        self.max_steps = int(env.get("max_steps", 220))
        self.origin = np.asarray(env.get("origin", [0.0, 0.0, -14.0]), dtype=float)
        self.x_min = float(env.get("x_min", -95.0))
        self.x_max = float(env.get("x_max", 95.0))
        self.y_min = float(env.get("y_min", -95.0))
        self.y_max = float(env.get("y_max", 95.0))
        self.default_z = float(env.get("default_z", self.origin[2] if self.origin.shape[0] >= 3 else 0.0))
        self.clamp_predicted_position = bool(env.get("clamp_predicted_position", False))
        self.enemy_step_safety = float(env.get("enemy_step_safety", 0.9))
        self.origin_radius_xy = float(env.get("origin_radius_xy", 2.5))

        self.enemy_speed = float(enemy.get("speed", 15.0))
        self.enemy_spawn_r = float(enemy.get("spawn_radius", 55.0))
        self.policy_mode = str(enemy.get("policy_mode", "mpc_lite"))

        self.step_count = 0
        self.overtime_assault_active = False
        self.enemy_prev_heading: Optional[float] = None
        self.enemy_prev_cmd_xy = np.zeros(2, dtype=float)
        self.enemy_dash_active = False
        self.enemy_block_latch = 0
        self.enemy_delay_buffer: deque[np.ndarray] = deque()

    @classmethod
    def from_file(cls, path: Union[Path, str] = DEFAULT_CONFIG_PATH) -> "EnemyStrategyMPC":
        return cls(_load_yaml(Path(path)))

    def reset(self) -> None:
        self.step_count = 0
        self.overtime_assault_active = False
        self.enemy_prev_heading = None
        self.enemy_prev_cmd_xy = np.zeros(2, dtype=float)
        self.enemy_dash_active = False
        self.enemy_block_latch = 0
        self.enemy_delay_buffer.clear()

    def plan(
        self,
        enemy_pos: Iterable[float],
        enemy_vel: Iterable[float],
        defender_positions: Iterable[Iterable[float]],
        step_count: Optional[int] = None,
        overtime_assault_active: bool = False,
    ) -> Dict[str, Any]:
        if step_count is None:
            self.step_count += 1
        else:
            self.step_count = int(step_count)
        self.overtime_assault_active = bool(overtime_assault_active)

        enemy_pos_arr = np.asarray(enemy_pos, dtype=float)
        enemy_vel_arr = np.asarray(enemy_vel, dtype=float)
        if enemy_pos_arr.shape[0] < 2 or enemy_vel_arr.shape[0] < 2:
            raise ValueError("enemy_pos and enemy_vel must contain at least x/y values")
        if enemy_pos_arr.shape[0] == 2:
            enemy_pos_arr = np.array([enemy_pos_arr[0], enemy_pos_arr[1], self.default_z], dtype=float)
        if enemy_vel_arr.shape[0] == 2:
            enemy_vel_arr = np.array([enemy_vel_arr[0], enemy_vel_arr[1], 0.0], dtype=float)

        def_xy = _as_xy_array(defender_positions)

        if self.policy_mode == "mpc_lite":
            vx, vy, yaw, debug = self._enemy_plan_velocity(enemy_pos_arr, enemy_vel_arr, def_xy)
        else:
            to_origin = self.origin[:2] - enemy_pos_arr[:2]
            dist_xy = float(np.linalg.norm(to_origin)) + 1e-9
            unit = to_origin / dist_xy
            v_cap = self.enemy_step_safety * (dist_xy / self.dt)
            speed = float(min(self.enemy_speed, v_cap))
            vx = float(unit[0] * speed)
            vy = float(unit[1] * speed)
            yaw = vec2_to_yaw(unit)
            debug = {"mode": "direct"}

        predicted = enemy_pos_arr.astype(float).copy()
        predicted[0] += vx * self.dt
        predicted[1] += vy * self.dt
        if self.clamp_predicted_position:
            predicted[0] = float(np.clip(predicted[0], self.x_min, self.x_max))
            predicted[1] = float(np.clip(predicted[1], self.y_min, self.y_max))

        return {
            "velocity_xy": [float(vx), float(vy)],
            "yaw_rad": float(yaw),
            "yaw_deg": float(np.degrees(yaw)),
            "predicted_position": predicted.tolist(),
            "predicted_xy": [float(predicted[0]), float(predicted[1])],
            "step_count": int(self.step_count),
            "debug": debug,
        }

    def _enemy_boundary_cost(self, xy: np.ndarray, margin: float) -> float:
        margin = max(1.0, float(margin))
        nx = max(0.0, abs(float(xy[0])) - (self.x_max - margin)) / margin
        ny = max(0.0, abs(float(xy[1])) - (self.y_max - margin)) / margin
        return float(min(2.0, nx + ny))

    def _enemy_surround_risk(self, enemy_xy: np.ndarray, def_xy: np.ndarray, radius: float) -> float:
        radius = max(1e-6, float(radius))
        dxy = def_xy - enemy_xy[None, :]
        d = np.linalg.norm(dxy, axis=1)
        mask = d < radius
        if not np.any(mask):
            return 0.0

        ang = np.sort(np.arctan2(dxy[mask, 1], dxy[mask, 0]))
        if ang.size <= 1:
            coverage = 0.12
        else:
            diffs = np.diff(np.concatenate([ang, [ang[0] + 2.0 * np.pi]]))
            max_gap = float(np.max(diffs))
            coverage = float(np.clip(1.0 - max_gap / (2.0 * np.pi), 0.0, 1.0))

        proximity = float(np.mean(np.exp(-d[mask] / (0.55 * radius + 1e-6))))
        return float(0.55 * coverage + 0.45 * proximity)

    def _enemy_front_blocked(self, enemy_xy: np.ndarray, def_xy: np.ndarray, cfg: Dict[str, Any]) -> bool:
        if def_xy is None or len(def_xy) == 0:
            return False

        to_origin = self.origin[:2] - enemy_xy
        dist_goal = float(np.linalg.norm(to_origin))
        if dist_goal < 1e-9:
            return False

        u_r = to_origin / (dist_goal + 1e-12)
        u_t = np.array([-u_r[1], u_r[0]], dtype=float)
        lookahead = min(float(cfg.get("arc_front_lookahead", 28.0)), dist_goal)
        lat_thr = float(cfg.get("arc_block_lateral", 9.0))
        dist_thr = float(cfg.get("arc_block_dist", 18.0))

        pre_lookahead = min(float(cfg.get("pre_block_lookahead", lookahead * 1.35)), dist_goal)
        pre_lat_thr = float(cfg.get("pre_block_lateral", lat_thr * 1.35))
        pre_dist_thr = float(cfg.get("pre_block_dist", dist_thr * 1.45))
        pre_min_proj = float(cfg.get("pre_block_min_proj", 6.0))

        for p in def_xy:
            rel = p - enemy_xy
            proj = float(np.dot(rel, u_r))
            lat = float(np.dot(rel, u_t))
            d = float(np.linalg.norm(rel))
            if (0.0 < proj < lookahead) and (abs(lat) < lat_thr) and (d < dist_thr):
                return True
            if (pre_min_proj < proj < pre_lookahead) and (abs(lat) < pre_lat_thr) and (d < pre_dist_thr):
                return True

        pressure = self._enemy_front_pressure(enemy_xy, def_xy, cfg)
        return bool(pressure >= float(cfg.get("pre_block_pressure_thr", 0.35)))

    def _enemy_front_pressure(self, enemy_xy: np.ndarray, def_xy: np.ndarray, cfg: Dict[str, Any]) -> float:
        if def_xy is None or len(def_xy) == 0:
            return 0.0

        to_origin = self.origin[:2] - enemy_xy
        dist_goal = float(np.linalg.norm(to_origin))
        if dist_goal < 1e-9:
            return 0.0

        u_r = to_origin / (dist_goal + 1e-12)
        u_t = np.array([-u_r[1], u_r[0]], dtype=float)
        lookahead = min(float(cfg.get("arc_front_lookahead", 28.0)), dist_goal)
        lat_thr = float(cfg.get("arc_block_lateral", 9.0))
        dist_thr = float(cfg.get("arc_block_dist", 18.0))

        pressure = 0.0
        for p in def_xy:
            rel = p - enemy_xy
            proj = float(np.dot(rel, u_r))
            lat = float(np.dot(rel, u_t))
            d = float(np.linalg.norm(rel))
            if proj <= 0.0:
                continue
            if proj >= lookahead or d >= dist_thr * 1.15:
                continue
            lane = max(0.0, 1.0 - abs(lat) / max(1e-6, lat_thr))
            depth = max(0.0, 1.0 - proj / max(1e-6, lookahead))
            near = max(0.0, 1.0 - d / max(1e-6, dist_thr * 1.15))
            pressure += lane * (0.55 * depth + 0.45 * near)

        return float(np.clip(pressure, 0.0, 2.5))

    def _enemy_goal_opening(self, enemy_xy: np.ndarray, def_xy: np.ndarray, cfg: Dict[str, Any]) -> bool:
        if def_xy is None or len(def_xy) == 0:
            return True

        to_origin = self.origin[:2] - enemy_xy
        dist_goal = float(np.linalg.norm(to_origin))
        if dist_goal < 1e-9:
            return True

        u_r = to_origin / (dist_goal + 1e-12)
        u_t = np.array([-u_r[1], u_r[0]], dtype=float)
        corridor_len = min(float(cfg.get("commit_clear_front_dist", 24.0)), dist_goal)
        lane_half = float(cfg.get("commit_lane_half_width", max(4.0, float(cfg.get("arc_block_lateral", 9.0)) * 0.72)))
        sep_min = float(cfg.get("commit_sep_dist", 8.5))
        pressure_thr = float(cfg.get("commit_pressure_thr", 0.20))

        blockers = 0
        nearest_lane = 1e9
        for p in def_xy:
            rel = p - enemy_xy
            proj = float(np.dot(rel, u_r))
            lat = abs(float(np.dot(rel, u_t)))
            d = float(np.linalg.norm(rel))
            if proj <= 0.0 or proj >= corridor_len:
                continue
            if lat < lane_half:
                blockers += 1
                nearest_lane = min(nearest_lane, d)
            elif lat < lane_half * 1.8 and d < sep_min * 1.1:
                nearest_lane = min(nearest_lane, d)

        pressure = self._enemy_front_pressure(enemy_xy, def_xy, cfg)
        clear_lane = (blockers == 0) or (nearest_lane > sep_min)
        return bool(clear_lane and pressure <= pressure_thr)

    def _enemy_arc_headings(self, enemy_xy: np.ndarray, def_xy: np.ndarray, cfg: Dict[str, Any]) -> List[float]:
        to_origin = self.origin[:2] - enemy_xy
        dist_goal = float(np.linalg.norm(to_origin))
        if dist_goal < 1e-9:
            return []

        goal_heading = vec2_to_yaw(to_origin)
        left_h = wrap_pi(goal_heading + np.pi / 2.0)
        right_h = wrap_pi(goal_heading - np.pi / 2.0)
        arc_offset = math.radians(float(cfg.get("arc_offset_deg", 18.0)))
        speed = float(min(self.enemy_speed * float(cfg["speed_scale"]), self.enemy_step_safety * (dist_goal / self.dt)))
        step = speed * self.dt

        def side_score(h: float) -> float:
            p1 = enemy_xy + yaw_to_unit(h) * step
            near_d = float(np.min(np.linalg.norm(def_xy - p1[None, :], axis=1))) if len(def_xy) > 0 else 1e9
            new_goal_d = float(np.linalg.norm(self.origin[:2] - p1))
            radial_pen = max(0.0, new_goal_d - dist_goal)
            wall = self._enemy_boundary_cost(p1, float(cfg.get("wall_margin", 18.0)))
            return near_d - 4.0 * radial_pen - 2.0 * wall

        scored_sides = [(side_score(left_h), left_h), (side_score(right_h), right_h)]
        scored_sides.sort(key=lambda x: x[0], reverse=True)

        heads = []
        for _score, h in scored_sides:
            heads.extend([h, wrap_pi(h + arc_offset), wrap_pi(h - arc_offset)])
        return heads

    def _enemy_candidate_headings(
        self,
        enemy_xy: np.ndarray,
        def_xy: np.ndarray,
        current_heading: float,
        cfg: Dict[str, Any],
        blocked: bool = False,
    ) -> List[float]:
        to_origin = self.origin[:2] - enemy_xy
        dist_goal = float(np.linalg.norm(to_origin))
        goal_heading = vec2_to_yaw(to_origin)
        spread = np.radians(float(cfg["heading_spread_deg"]))
        n_cand = int(cfg["candidate_count"])

        radial_min_cos = float(cfg.get("min_radial_progress_cos", -1.0))
        heads = []
        if blocked:
            radial_min_cos = max(0.18, radial_min_cos)
            heads.extend(self._enemy_arc_headings(enemy_xy, def_xy, cfg))
            tangent_left = wrap_pi(goal_heading + math.radians(78.0))
            tangent_right = wrap_pi(goal_heading - math.radians(78.0))
            heads.extend([
                tangent_left, tangent_right,
                wrap_pi(tangent_left + math.radians(14.0)), wrap_pi(tangent_left - math.radians(14.0)),
                wrap_pi(tangent_right + math.radians(14.0)), wrap_pi(tangent_right - math.radians(14.0)),
                wrap_pi(goal_heading + math.radians(48.0)), wrap_pi(goal_heading - math.radians(48.0)),
            ])
        else:
            heads = [goal_heading, current_heading]
            for delta in np.linspace(-spread, spread, n_cand):
                heads.append(wrap_pi(goal_heading + float(delta)))
            d = np.linalg.norm(def_xy - enemy_xy[None, :], axis=1)
            if d.size > 0:
                j = int(np.argmin(d))
                away_vec = enemy_xy - def_xy[j]
                if float(np.linalg.norm(away_vec)) > 1e-9:
                    away_h = vec2_to_yaw(away_vec)
                    heads.extend([wrap_pi(away_h + math.radians(55.0)), wrap_pi(away_h - math.radians(55.0))])

        uniq = []
        seen = set()
        u_goal = to_origin / (dist_goal + 1e-12)
        for h in heads:
            h = wrap_pi(h)
            move_u = yaw_to_unit(h)
            proj_cos = float(np.dot(move_u, u_goal))
            if proj_cos < radial_min_cos:
                continue
            if blocked and proj_cos > 0.72:
                continue
            key = round(h, 4)
            if key not in seen:
                seen.add(key)
                uniq.append(float(key))

        if not uniq:
            uniq = self._enemy_arc_headings(enemy_xy, def_xy, cfg) if blocked else [goal_heading]
            if not uniq:
                uniq = [goal_heading]
        return uniq

    def _enemy_rollout_cost(
        self,
        enemy_xy: np.ndarray,
        def_xy: np.ndarray,
        heading: float,
        current_heading: float,
        cfg: Dict[str, Any],
    ) -> float:
        pos = np.array(enemy_xy, dtype=float)
        yaw = float(heading)
        total = 0.0
        discount = float(cfg["rollout_discount"])
        spawn_scale = max(1.0, float(self.enemy_spawn_r))

        for k in range(int(cfg["horizon"])):
            to_origin = self.origin[:2] - pos
            dist_goal = float(np.linalg.norm(to_origin))
            if dist_goal < 1e-9:
                break

            blocked_now = self._enemy_front_blocked(pos, def_xy, cfg)
            goal_w = float(cfg["goal_w"]) * (float(cfg.get("blocked_goal_w_mult", 1.0)) if blocked_now else 1.0)
            sep_w = float(cfg["sep_w"]) * (float(cfg.get("blocked_sep_w_mult", 1.0)) if blocked_now else 1.0)
            surround_w = float(cfg["surround_w"]) * (float(cfg.get("blocked_surround_w_mult", 1.0)) if blocked_now else 1.0)

            v_cap = self.enemy_step_safety * (dist_goal / self.dt)
            speed_scale = float(cfg.get("blocked_speed_scale", cfg["speed_scale"])) if blocked_now else float(cfg["speed_scale"])
            speed = float(min(self.enemy_speed * speed_scale, v_cap))
            step_vec = yaw_to_unit(yaw) * speed * self.dt
            next_pos = pos + step_vec

            next_dist_goal = float(np.linalg.norm(self.origin[:2] - next_pos))
            near_d = float(np.min(np.linalg.norm(def_xy - next_pos[None, :], axis=1))) if len(def_xy) > 0 else 1e9
            goal_cost = next_dist_goal / spawn_scale
            sep_cost = float(np.exp(-near_d / max(1e-6, float(cfg["sep_scale"]))))
            surround_cost = self._enemy_surround_risk(next_pos, def_xy, float(cfg["surround_radius"]))
            wall_cost = self._enemy_boundary_cost(next_pos, float(cfg["wall_margin"]))
            front_pressure = self._enemy_front_pressure(next_pos, def_xy, cfg)
            turn_cost = abs(wrap_pi(yaw - current_heading)) / np.pi if k == 0 else 0.0

            radial_backtrack = max(0.0, next_dist_goal - dist_goal) / spawn_scale
            progress_gain = max(0.0, dist_goal - next_dist_goal) / spawn_scale
            step_cost = (
                goal_w * goal_cost
                + sep_w * sep_cost
                + surround_w * surround_cost
                + float(cfg["wall_w"]) * wall_cost
                + float(cfg["turn_w"]) * turn_cost
                + float(cfg.get("radial_backtrack_w", 2.0)) * radial_backtrack
                + float(cfg.get("front_pressure_w", 0.0)) * front_pressure
                - 0.45 * goal_w * progress_gain
            )
            total += (discount ** k) * step_cost

            pos = next_pos
            goal_heading = vec2_to_yaw(self.origin[:2] - pos)
            yaw = wrap_pi((1.0 - float(cfg["retarget_blend"])) * yaw + float(cfg["retarget_blend"]) * goal_heading)

        return float(total)

    def _enemy_should_dash(self, enemy_xy: np.ndarray, def_xy: np.ndarray, cfg: Dict[str, Any]) -> bool:
        d_enemy = float(np.linalg.norm(self.origin[:2] - enemy_xy))
        d_def_min = float(np.min(np.linalg.norm(def_xy - self.origin[:2][None, :], axis=1)))
        margin = max(0.0, float(cfg.get("dash_closest_margin", 0.0)))
        return bool(d_enemy + margin < d_def_min)

    def _enemy_time_commit_alpha(self) -> float:
        start_frac = float(self.cfg.get("time_commit_start_frac", 0.50))
        start_frac = float(np.clip(start_frac, 0.0, 0.98))
        prog = float(self.step_count) / float(max(1, int(self.max_steps)))
        if prog <= start_frac:
            return 0.0
        return float(np.clip((prog - start_frac) / max(1e-6, 1.0 - start_frac), 0.0, 1.0))

    def _enemy_plan_velocity(
        self,
        enemy_pos: np.ndarray,
        enemy_vel: np.ndarray,
        def_xy: np.ndarray,
    ) -> Tuple[float, float, float, Dict[str, Any]]:
        cfg = self.cfg
        enemy_xy = enemy_pos[:2].astype(float)

        if self.enemy_prev_heading is None:
            if float(np.linalg.norm(enemy_vel[:2])) > 1e-6:
                current_heading = vec2_to_yaw(enemy_vel[:2])
            else:
                current_heading = vec2_to_yaw(self.origin[:2] - enemy_xy)
        else:
            current_heading = float(self.enemy_prev_heading)

        to_origin = self.origin[:2] - enemy_xy
        dist_xy = float(np.linalg.norm(to_origin))
        goal_heading = vec2_to_yaw(to_origin)
        time_commit_alpha = self._enemy_time_commit_alpha()
        prog = float(self.step_count) / float(max(1, int(self.max_steps)))
        hard_commit_active = bool(self.overtime_assault_active or (prog >= float(cfg.get("time_hard_commit_start_frac", 0.80))))
        remaining_steps = max(0, int(self.max_steps) - int(self.step_count))
        remaining_time = max(self.dt, float(remaining_steps) * self.dt)
        final_goal_window = float(cfg.get("time_force_goal_window_sec", 1.0))
        force_direct_commit = bool(self.overtime_assault_active or (time_commit_alpha > 1e-9 and remaining_time <= final_goal_window))

        blocked_now = self._enemy_front_blocked(enemy_xy, def_xy, cfg)
        opening_now = self._enemy_goal_opening(enemy_xy, def_xy, cfg)
        if blocked_now and (not opening_now):
            self.enemy_block_latch = int(max(self.enemy_block_latch, int(cfg.get("block_latch_steps", 0))))
        elif opening_now:
            self.enemy_block_latch = 0
        else:
            self.enemy_block_latch = max(0, int(self.enemy_block_latch) - 1)
        blocked = bool((blocked_now or (self.enemy_block_latch > 0)) and (not opening_now))

        dash_mode = self._enemy_should_dash(enemy_xy, def_xy, cfg)
        opportunity_mode = bool((not dash_mode) and opening_now and (dist_xy < float(cfg.get("commit_dist", 18.0))))
        mode = "normal"

        def apply_time_commit(plan_cfg: Dict[str, Any]) -> Dict[str, Any]:
            if time_commit_alpha <= 1e-9:
                return plan_cfg
            plan_cfg = dict(plan_cfg)
            plan_cfg["goal_w"] = float(plan_cfg.get("goal_w", 1.0)) * lerp(1.0, cfg.get("time_goal_w_mult_end", 2.10), time_commit_alpha)
            plan_cfg["sep_w"] = float(plan_cfg.get("sep_w", 0.0)) * lerp(1.0, cfg.get("time_sep_w_mult_end", 0.18), time_commit_alpha)
            plan_cfg["surround_w"] = float(plan_cfg.get("surround_w", 0.0)) * lerp(1.0, cfg.get("time_surround_w_mult_end", 0.24), time_commit_alpha)
            if "front_pressure_w" in plan_cfg:
                plan_cfg["front_pressure_w"] = float(plan_cfg.get("front_pressure_w", 0.0)) * lerp(1.0, cfg.get("time_front_pressure_w_mult_end", 0.18), time_commit_alpha)
            plan_cfg["turn_w"] = float(plan_cfg.get("turn_w", 0.0)) * lerp(1.0, cfg.get("time_turn_w_mult_end", 0.30), time_commit_alpha)
            plan_cfg["blocked_goal_w_mult"] = float(plan_cfg.get("blocked_goal_w_mult", 1.0)) * lerp(1.0, cfg.get("time_blocked_goal_w_mult_end", 2.10), time_commit_alpha)
            plan_cfg["blocked_sep_w_mult"] = float(plan_cfg.get("blocked_sep_w_mult", 1.0)) * lerp(1.0, cfg.get("time_blocked_sep_w_mult_end", 0.26), time_commit_alpha)
            plan_cfg["blocked_surround_w_mult"] = float(plan_cfg.get("blocked_surround_w_mult", 1.0)) * lerp(1.0, cfg.get("time_blocked_surround_w_mult_end", 0.32), time_commit_alpha)
            plan_cfg["reaction_steps"] = int(round(float(plan_cfg.get("reaction_steps", 0)) * (1.0 - time_commit_alpha)))
            plan_cfg["eps_random"] = lerp(float(plan_cfg.get("eps_random", 0.0)), float(cfg.get("time_eps_random_end", 0.0)), time_commit_alpha)
            plan_cfg["jitter_deg"] = lerp(float(plan_cfg.get("jitter_deg", 0.0)), float(cfg.get("time_jitter_deg_end", 0.0)), time_commit_alpha)
            return plan_cfg

        if hard_commit_active:
            mode = "hard_commit"
            hard_half = math.radians(float(cfg.get("time_hard_commit_goal_half_deg", 10.0)))
            n_hard = max(3, int(cfg.get("time_hard_commit_candidate_count", 5)))
            candidates = [goal_heading]
            for delta in np.linspace(-hard_half, hard_half, n_hard):
                candidates.append(wrap_pi(goal_heading + float(delta)))
            if not self.overtime_assault_active:
                side_slip = math.radians(max(6.0, 1.35 * float(cfg.get("time_hard_commit_goal_half_deg", 10.0))))
                candidates.extend([wrap_pi(goal_heading + side_slip), wrap_pi(goal_heading - side_slip)])

            plan_cfg = dict(cfg)
            plan_cfg["goal_w"] = float(cfg.get("goal_w", 1.0)) * float(cfg.get("time_hard_commit_goal_w_mult", 3.20))
            plan_cfg["sep_w"] = float(cfg.get("sep_w", 0.0)) * float(cfg.get("time_hard_commit_sep_w_mult", 0.10))
            plan_cfg["surround_w"] = float(cfg.get("surround_w", 0.0)) * float(cfg.get("time_hard_commit_surround_w_mult", 0.10))
            plan_cfg["front_pressure_w"] = float(cfg.get("front_pressure_w", 0.0)) * float(cfg.get("time_hard_commit_front_pressure_w_mult", 0.08))
            plan_cfg["turn_w"] = float(cfg.get("turn_w", 0.0)) * float(cfg.get("time_hard_commit_turn_w_mult", 0.05))
            plan_cfg["blocked_goal_w_mult"] = 1.0
            plan_cfg["blocked_sep_w_mult"] = max(0.02, float(cfg.get("time_hard_commit_sep_w_mult", 0.10)))
            plan_cfg["blocked_surround_w_mult"] = max(0.02, float(cfg.get("time_hard_commit_surround_w_mult", 0.10)))
            plan_cfg["reaction_steps"] = 0
            plan_cfg["eps_random"] = 0.0
            plan_cfg["jitter_deg"] = 0.0

            scored = [(self._enemy_rollout_cost(enemy_xy, def_xy, h, current_heading, plan_cfg), h) for h in candidates]
            scored.sort(key=lambda x: x[0])
            chosen_heading = float(scored[0][1]) if scored else goal_heading

            if self.overtime_assault_active:
                chosen_heading = goal_heading
            else:
                mix = float(np.clip(cfg.get("time_hard_commit_heading_mix", 0.95), 0.0, 1.0))
                mix_vec = (1.0 - mix) * yaw_to_unit(chosen_heading) + mix * yaw_to_unit(goal_heading)
                chosen_heading = vec2_to_yaw(mix_vec) if float(np.linalg.norm(mix_vec)) > 1e-8 else goal_heading

            speed_scale = max(float(cfg.get("time_hard_commit_speed_scale", 1.18)), float(cfg.get("speed_scale", 1.0)))
            speed_scale *= lerp(1.0, cfg.get("time_speed_scale_end", 1.10), time_commit_alpha)
            if self.overtime_assault_active:
                speed_scale = max(speed_scale, float(cfg.get("overtime_speed_scale", 1.28)))
            delay = 0
        elif dash_mode:
            mode = "dash"
            chosen_heading = goal_heading
            jitter = lerp(float(cfg.get("dash_jitter_deg", 0.0)), 0.0, time_commit_alpha)
            if (not force_direct_commit) and jitter > 1e-6:
                chosen_heading = wrap_pi(chosen_heading + math.radians(float(np.random.uniform(-jitter, jitter))))
            speed_scale = float(cfg.get("dash_speed_scale", cfg["speed_scale"])) * lerp(1.0, cfg.get("time_speed_scale_end", 1.10), time_commit_alpha)
            delay = int(round(float(cfg.get("dash_reaction_steps", 0)) * (1.0 - time_commit_alpha)))
        else:
            plan_cfg = cfg
            if opportunity_mode:
                mode = "opportunity"
                half = math.radians(float(cfg.get("commit_goal_half_deg", 24.0)))
                n_commit = int(cfg.get("commit_candidate_count", 7))
                candidates = [current_heading]
                for delta in np.linspace(-half, half, n_commit):
                    candidates.append(wrap_pi(goal_heading + float(delta)))
                plan_cfg = dict(cfg)
                plan_cfg["goal_w"] = float(cfg["goal_w"]) * float(cfg.get("commit_goal_w_mult", 1.30))
                plan_cfg["sep_w"] = float(cfg["sep_w"]) * float(cfg.get("commit_sep_w_mult", 0.85))
                plan_cfg["surround_w"] = float(cfg["surround_w"]) * float(cfg.get("commit_surround_w_mult", 0.90))
                plan_cfg["front_pressure_w"] = float(cfg.get("front_pressure_w", 0.0)) * float(cfg.get("commit_front_pressure_mult", 0.55))
                plan_cfg["blocked_goal_w_mult"] = 1.0
                plan_cfg["blocked_sep_w_mult"] = 1.0
                plan_cfg["blocked_surround_w_mult"] = 1.0
                plan_cfg["reaction_steps"] = 0
                plan_cfg["jitter_deg"] = float(cfg["jitter_deg"]) * 0.35
            elif blocked:
                mode = "blocked"
                candidates = self._enemy_candidate_headings(enemy_xy, def_xy, current_heading, cfg, blocked=True)
                uniq = []
                seen = set()
                for h in candidates:
                    key = round(wrap_pi(h), 4)
                    if key not in seen:
                        seen.add(key)
                        uniq.append(float(key))
                candidates = uniq
            else:
                candidates = self._enemy_candidate_headings(enemy_xy, def_xy, current_heading, cfg, blocked=False)

            plan_cfg = apply_time_commit(plan_cfg)
            scored = [(self._enemy_rollout_cost(enemy_xy, def_xy, h, current_heading, plan_cfg), h) for h in candidates]
            scored.sort(key=lambda x: x[0])

            if (not blocked) and (not opportunity_mode) and random.random() < float(plan_cfg.get("eps_random", 0.0)):
                chosen_heading = float(random.choice(candidates))
            else:
                chosen_heading = float(scored[0][1])

            heading_mix = float(np.clip(0.10 + 0.90 * time_commit_alpha, 0.0, 1.0))
            if force_direct_commit:
                heading_mix = 1.0
            if heading_mix > 1e-6:
                mix_vec = (1.0 - heading_mix) * yaw_to_unit(chosen_heading) + heading_mix * yaw_to_unit(goal_heading)
                chosen_heading = vec2_to_yaw(mix_vec) if float(np.linalg.norm(mix_vec)) > 1e-8 else goal_heading

            jitter = float(plan_cfg.get("jitter_deg", cfg.get("jitter_deg", 0.0)))
            if blocked:
                jitter *= 0.20
            if (not force_direct_commit) and jitter > 1e-6:
                chosen_heading = wrap_pi(chosen_heading + math.radians(float(np.random.uniform(-jitter, jitter))))
            if opportunity_mode:
                speed_scale = max(float(plan_cfg.get("commit_speed_scale", plan_cfg["speed_scale"])), float(plan_cfg["speed_scale"]))
            else:
                speed_scale = float(plan_cfg.get("blocked_speed_scale", plan_cfg["speed_scale"])) if blocked else float(plan_cfg["speed_scale"])
            speed_scale *= lerp(1.0, cfg.get("time_speed_scale_end", 1.10), time_commit_alpha)
            delay = int(plan_cfg.get("reaction_steps", cfg.get("reaction_steps", 0)))

        v_cap = self.enemy_step_safety * (dist_xy / self.dt)
        speed = float(min(self.enemy_speed * speed_scale, v_cap))
        near_goal_aim_dist = float(cfg.get("near_goal_aim_dist", 8.0))
        if dist_xy < near_goal_aim_dist:
            near_goal_scale = lerp(float(cfg.get("near_goal_speed_scale", 0.84)), float(cfg.get("time_near_goal_speed_scale_end", 0.96)), time_commit_alpha)
            speed *= near_goal_scale
            terminal_gain = lerp(0.34, float(cfg.get("time_terminal_cap_gain_end", 0.78)), time_commit_alpha)
            terminal_cap = max(1.4, terminal_gain * (dist_xy + self.origin_radius_xy) / self.dt)
            speed = min(speed, terminal_cap)

        if force_direct_commit:
            mode = "force_direct_commit"
            chosen_heading = goal_heading
            commit_mult = float(cfg.get("overtime_speed_scale", 1.28)) if self.overtime_assault_active else float(cfg.get("time_final_second_speed_scale", 1.35))
            max_commit_speed = min(v_cap, float(self.enemy_speed) * commit_mult)
            need_speed = float(cfg.get("time_arrival_margin", 0.92)) * dist_xy / max(remaining_time, self.dt)
            speed = min(max_commit_speed, max(speed, need_speed))
            delay = 0

        cmd_xy = yaw_to_unit(chosen_heading) * speed

        if force_direct_commit:
            self.enemy_delay_buffer.clear()
            out_xy = np.array(cmd_xy, dtype=float)
            self.enemy_dash_active = dash_mode
        else:
            if dash_mode != self.enemy_dash_active:
                self.enemy_delay_buffer.clear()
                self.enemy_dash_active = dash_mode

            self.enemy_delay_buffer.append(np.array(cmd_xy, dtype=float))
            if len(self.enemy_delay_buffer) > delay:
                out_xy = np.array(self.enemy_delay_buffer.popleft(), dtype=float)
            else:
                out_xy = np.array(self.enemy_prev_cmd_xy if np.linalg.norm(self.enemy_prev_cmd_xy) > 1e-6 else cmd_xy, dtype=float)

        if ((dash_mode and (not blocked)) or force_direct_commit) and dist_xy < near_goal_aim_dist:
            out_xy = yaw_to_unit(goal_heading) * min(
                np.linalg.norm(out_xy),
                max(1.4, lerp(0.30, 0.88, time_commit_alpha) * (dist_xy + self.origin_radius_xy) / self.dt),
            )

        self.enemy_prev_cmd_xy = out_xy.copy()
        self.enemy_prev_heading = vec2_to_yaw(out_xy if np.linalg.norm(out_xy) > 1e-6 else cmd_xy)
        debug = {
            "mode": mode,
            "blocked": bool(blocked),
            "blocked_now": bool(blocked_now),
            "opening_now": bool(opening_now),
            "dash_mode": bool(dash_mode),
            "opportunity_mode": bool(opportunity_mode),
            "time_commit_alpha": float(time_commit_alpha),
            "dist_to_origin": float(dist_xy),
            "delay": int(delay),
            "speed": float(np.linalg.norm(out_xy)),
        }
        return float(out_xy[0]), float(out_xy[1]), float(self.enemy_prev_heading), debug


def load_enemy_strategy(config_path: Union[Path, str] = DEFAULT_CONFIG_PATH) -> EnemyStrategyMPC:
    return EnemyStrategyMPC.from_file(config_path)
