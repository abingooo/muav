#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import numpy as np

from adv_module.game_end import GameEndConfig, GameEndMonitor
from mpc_module.game_end import GameEndConfig as MpcGameEndConfig
from mpc_module.game_end import GameEndMonitor as MpcGameEndMonitor


def _pos(x, y, z=1.0):
    return np.asarray([x, y, z], dtype=np.float32)


class AdvGameEndTest(unittest.TestCase):
    def _monitor(self):
        return GameEndMonitor(
            GameEndConfig(
                enabled=True,
                capture_distance_m=1.0,
                asset_distance_m=1.0,
                hold_duration_sec=0.5,
                asset_origin=_pos(0.0, 0.0),
                x_bounds=(-5.0, 5.0),
                y_bounds=(-10.0, 10.0),
            )
        )

    def test_defender_capture_requires_hold_duration(self):
        monitor = self._monitor()
        positions = {
            "enemy": _pos(1.0, 1.0),
            "defender_0": _pos(1.4, 1.0),
        }

        self.assertFalse(monitor.update(positions, 10.0).active)
        self.assertFalse(monitor.update(positions, 10.49).active)
        status = monitor.update(positions, 10.5)

        self.assertTrue(status.active)
        self.assertEqual(status.outcome, "defender_win")
        self.assertEqual(status.trigger_roles, ("defender_0", "enemy"))

    def test_enemy_asset_entry_uses_horizontal_distance_and_hold_duration(self):
        monitor = self._monitor()
        positions = {
            "enemy": _pos(0.8, 0.0, 6.0),
            "defender_0": _pos(4.0, 4.0),
        }

        self.assertFalse(monitor.update(positions, 20.0).active)
        status = monitor.update(positions, 20.5)

        self.assertTrue(status.active)
        self.assertEqual(status.outcome, "enemy_win")
        self.assertEqual(status.trigger_roles, ("enemy",))

    def test_any_aircraft_out_of_bounds_fails_immediately(self):
        monitor = self._monitor()
        positions = {
            "enemy": _pos(0.0, 0.0),
            "defender_2": _pos(5.1, 0.0),
        }

        status = monitor.update(positions, 30.0)

        self.assertTrue(status.active)
        self.assertEqual(status.outcome, "experiment_failed")
        self.assertEqual(status.trigger_roles, ("defender_2",))

    def test_timers_reset_when_condition_breaks(self):
        monitor = self._monitor()
        near = {
            "enemy": _pos(2.0, 2.0),
            "defender_0": _pos(2.5, 2.0),
        }
        far = {
            "enemy": _pos(2.0, 2.0),
            "defender_0": _pos(4.0, 2.0),
        }

        self.assertFalse(monitor.update(near, 40.0).active)
        self.assertFalse(monitor.update(far, 40.3).active)
        self.assertFalse(monitor.update(near, 40.6).active)
        status = monitor.update(near, 41.1)

        self.assertTrue(status.active)
        self.assertEqual(status.outcome, "defender_win")


class MpcGameEndTest(AdvGameEndTest):
    def _monitor(self):
        return MpcGameEndMonitor(
            MpcGameEndConfig(
                enabled=True,
                capture_distance_m=1.0,
                asset_distance_m=1.0,
                hold_duration_sec=0.5,
                asset_origin=_pos(0.0, 0.0),
                x_bounds=(-5.0, 5.0),
                y_bounds=(-10.0, 10.0),
            )
        )


if __name__ == "__main__":
    unittest.main()
