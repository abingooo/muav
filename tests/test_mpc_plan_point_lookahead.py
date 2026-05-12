#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import numpy as np

from mpc_module.ros_adapter import _build_lookahead_plan_point_world


class MpcPlanPointLookaheadTest(unittest.TestCase):
    def test_uses_velocity_direction_and_configured_distance(self):
        current = np.asarray([-8.1, 0.0, 0.0], dtype=np.float32)
        predicted = np.asarray([-7.94, 0.0, 1.0], dtype=np.float32)
        velocity = np.asarray([0.8, 0.0, 0.0], dtype=np.float32)

        point = _build_lookahead_plan_point_world(
            current,
            predicted,
            velocity,
            lookahead_m=1.0,
            output_height=1.0,
            environment={"x_min": -8.2, "x_max": 4.5, "y_min": -4.5, "y_max": 4.5},
        )

        np.testing.assert_allclose(point, np.asarray([-7.1, 0.0, 1.0], dtype=np.float32), atol=1e-6)

    def test_zero_lookahead_keeps_predicted_position_behavior(self):
        current = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        predicted = np.asarray([0.16, 0.0, 1.0], dtype=np.float32)
        velocity = np.asarray([0.8, 0.0, 0.0], dtype=np.float32)

        point = _build_lookahead_plan_point_world(
            current,
            predicted,
            velocity,
            lookahead_m=0.0,
            output_height=1.0,
            environment={},
        )

        np.testing.assert_allclose(point, predicted, atol=1e-6)

    def test_clamps_to_configured_world_bounds(self):
        current = np.asarray([-8.1, 0.0, 0.0], dtype=np.float32)
        predicted = np.asarray([-8.18, 0.0, 1.0], dtype=np.float32)
        velocity = np.asarray([-0.8, 0.0, 0.0], dtype=np.float32)

        point = _build_lookahead_plan_point_world(
            current,
            predicted,
            velocity,
            lookahead_m=1.0,
            output_height=1.0,
            environment={"x_min": -8.2, "x_max": 4.5, "y_min": -4.5, "y_max": 4.5},
        )

        np.testing.assert_allclose(point, np.asarray([-8.2, 0.0, 1.0], dtype=np.float32), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
