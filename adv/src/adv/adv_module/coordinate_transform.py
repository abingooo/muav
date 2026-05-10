#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CoordinateTransform:
    translation: np.ndarray
    yaw_rad: float

    @property
    def rotation(self) -> np.ndarray:
        c = math.cos(self.yaw_rad)
        s = math.sin(self.yaw_rad)
        return np.asarray(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def local_to_world_position(self, position: np.ndarray) -> np.ndarray:
        return (self.rotation.dot(position.astype(np.float32)) + self.translation).astype(np.float32)

    def local_to_world_velocity(self, velocity: np.ndarray) -> np.ndarray:
        return self.rotation.dot(velocity.astype(np.float32)).astype(np.float32)

    def world_to_local_position(self, position: np.ndarray) -> np.ndarray:
        return self.rotation.T.dot(position.astype(np.float32) - self.translation).astype(np.float32)

    def world_to_local_velocity(self, velocity: np.ndarray) -> np.ndarray:
        return self.rotation.T.dot(velocity.astype(np.float32)).astype(np.float32)
