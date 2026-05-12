#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import numpy as np

try:
    import torch  # noqa: F401
except ImportError:
    torch = None

if torch is not None:
    from adv_module.ros_adapter import _build_lookahead_plan_point_world


@unittest.skipIf(torch is None, "adv_module.ros_adapter requires torch")
class AdvPlanPointLookaheadTest(unittest.TestCase):
    def test_uses_velocity_direction_and_configured_distance(self):
        current = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        predicted = np.asarray([0.1, 0.0, 1.0], dtype=np.float32)
        velocity = np.asarray([2.0, 0.0, 0.0], dtype=np.float32)

        point = _build_lookahead_plan_point_world(
            current,
            predicted,
            velocity,
            lookahead_m=1.0,
            output_height=1.0,
            x_bounds=(-8.1, 4.5),
            y_bounds=(-4.5, 4.5),
        )

        np.testing.assert_allclose(point, np.asarray([1.0, 0.0, 1.0], dtype=np.float32), atol=1e-6)

    def test_zero_lookahead_keeps_predicted_position_behavior(self):
        current = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        predicted = np.asarray([0.1, 0.0, 1.0], dtype=np.float32)
        velocity = np.asarray([2.0, 0.0, 0.0], dtype=np.float32)

        point = _build_lookahead_plan_point_world(
            current,
            predicted,
            velocity,
            lookahead_m=0.0,
            output_height=1.0,
            x_bounds=(-8.1, 4.5),
            y_bounds=(-4.5, 4.5),
        )

        np.testing.assert_allclose(point, predicted, atol=1e-6)

    def test_clamps_to_configured_world_bounds(self):
        current = np.asarray([-8.1, 0.0, 1.0], dtype=np.float32)
        predicted = np.asarray([-8.2, 0.0, 1.0], dtype=np.float32)
        velocity = np.asarray([-2.0, 0.0, 0.0], dtype=np.float32)

        point = _build_lookahead_plan_point_world(
            current,
            predicted,
            velocity,
            lookahead_m=1.0,
            output_height=1.0,
            x_bounds=(-8.1, 4.5),
            y_bounds=(-4.5, 4.5),
        )

        np.testing.assert_allclose(point, np.asarray([-8.1, 0.0, 1.0], dtype=np.float32), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
