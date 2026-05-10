#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import unittest

import numpy as np

from adv_module.coordinate_transform import CoordinateTransform as AdvCoordinateTransform
from mpc_module.coordinate_transform import CoordinateTransform as MpcCoordinateTransform


TRANSFORM_CLASSES = (
    ("adv", AdvCoordinateTransform),
    ("mpc", MpcCoordinateTransform),
)


class CoordinateTransformTest(unittest.TestCase):
    def test_local_world_position_round_trip(self):
        local_position = np.asarray([4.0, -2.5, 1.2], dtype=np.float32)
        for name, transform_cls in TRANSFORM_CLASSES:
            with self.subTest(package=name):
                transform = transform_cls(
                    translation=np.asarray([10.0, -3.0, 0.7], dtype=np.float32),
                    yaw_rad=math.radians(33.0),
                )

                world_position = transform.local_to_world_position(local_position)
                round_trip_position = transform.world_to_local_position(world_position)

                np.testing.assert_allclose(round_trip_position, local_position, rtol=1e-6, atol=1e-6)

    def test_local_world_velocity_round_trip(self):
        local_velocity = np.asarray([1.5, -0.75, 0.2], dtype=np.float32)
        for name, transform_cls in TRANSFORM_CLASSES:
            with self.subTest(package=name):
                transform = transform_cls(
                    translation=np.asarray([20.0, 30.0, 40.0], dtype=np.float32),
                    yaw_rad=math.radians(-47.0),
                )

                world_velocity = transform.local_to_world_velocity(local_velocity)
                round_trip_velocity = transform.world_to_local_velocity(world_velocity)

                np.testing.assert_allclose(round_trip_velocity, local_velocity, rtol=1e-6, atol=1e-6)

    def test_yaw_180_direction(self):
        local_position = np.asarray([2.0, -3.0, 1.0], dtype=np.float32)
        local_velocity = np.asarray([0.5, -1.0, 0.25], dtype=np.float32)
        expected_world_position = np.asarray([-2.0, 4.5, 1.0], dtype=np.float32)
        expected_world_velocity = np.asarray([-0.5, 1.0, 0.25], dtype=np.float32)

        for name, transform_cls in TRANSFORM_CLASSES:
            with self.subTest(package=name):
                transform = transform_cls(
                    translation=np.asarray([0.0, 1.5, 0.0], dtype=np.float32),
                    yaw_rad=math.radians(180.0),
                )

                np.testing.assert_allclose(
                    transform.local_to_world_position(local_position),
                    expected_world_position,
                    rtol=1e-6,
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    transform.local_to_world_velocity(local_velocity),
                    expected_world_velocity,
                    rtol=1e-6,
                    atol=1e-6,
                )

    def test_translation_does_not_affect_velocity(self):
        local_velocity = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        for name, transform_cls in TRANSFORM_CLASSES:
            with self.subTest(package=name):
                transform = transform_cls(
                    translation=np.asarray([100.0, -50.0, 8.0], dtype=np.float32),
                    yaw_rad=0.0,
                )

                np.testing.assert_allclose(
                    transform.local_to_world_velocity(local_velocity),
                    local_velocity,
                    rtol=1e-6,
                    atol=1e-6,
                )


if __name__ == "__main__":
    unittest.main()
