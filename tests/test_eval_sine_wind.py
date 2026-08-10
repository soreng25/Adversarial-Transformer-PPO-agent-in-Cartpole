import os
import tempfile
import unittest

import numpy as np

from eval_sine_wind import (
    SineTrial,
    additional_seeds,
    load_failure_trace,
    make_sine_trace,
    run_amplitude_sweep,
    save_failure_trace,
)


class SineTraceTests(unittest.TestCase):
    def test_trace_has_fixed_phase_zero_length_and_bounds(self):
        trace = make_sine_trace(
            amplitude=1.0,
            frequency=0.23,
            horizon=500,
            max_wind=1.0,
        )

        self.assertEqual(trace.shape, (500,))
        self.assertEqual(trace[0], 0.0)
        self.assertLessEqual(float(np.max(np.abs(trace))), 1.0)
        expected = np.sin(2.0 * np.pi * 0.23 * np.arange(500))
        np.testing.assert_allclose(trace, expected, rtol=0.0, atol=1e-14)

    def test_invalid_trace_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            make_sine_trace(-0.1)
        with self.assertRaises(ValueError):
            make_sine_trace(1.1, max_wind=1.0)
        with self.assertRaises(ValueError):
            make_sine_trace(0.5, frequency=0.0)
        with self.assertRaises(ValueError):
            make_sine_trace(0.5, horizon=0)


class AmplitudeSweepTests(unittest.TestCase):
    def test_all_amplitudes_run_and_lowest_failure_is_selected(self):
        calls = []

        def replay(trace, seed):
            amplitude = float(np.max(np.abs(trace)))
            calls.append((amplitude, seed))
            return amplitude > 0.45, 123 if amplitude > 0.45 else len(trace)

        trials, failure = run_amplitude_sweep(
            replay,
            amplitudes=(0.2, 0.5, 0.8),
            horizon=500,
            seed=1006,
        )

        self.assertEqual(len(calls), 3)
        self.assertEqual(len(trials), 3)
        self.assertIsNotNone(failure)
        self.assertEqual(failure.amplitude, 0.5)
        self.assertEqual(failure.failure_step, 123)
        self.assertTrue(all(seed == 1006 for _, seed in calls))

    def test_no_failure_returns_none(self):
        trials, failure = run_amplitude_sweep(
            lambda trace, seed: (False, len(trace)),
            amplitudes=(0.1, 0.2),
        )

        self.assertEqual(len(trials), 2)
        self.assertIsNone(failure)


class SavedTraceTests(unittest.TestCase):
    def test_saved_trace_loads_exactly(self):
        trace = make_sine_trace(0.7)
        trial = SineTrial(
            amplitude=0.7,
            failed=True,
            failure_step=211,
            trace=trace,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "failure.npz")
            save_failure_trace(
                path,
                trial,
                frequency=0.23,
                seed=1006,
                horizon=500,
                max_wind=1.0,
                victim_checkpoint="checkpoints/victim",
                victim_env="cartpole",
            )
            saved = load_failure_trace(path)

        np.testing.assert_array_equal(saved["winds"], trace)
        self.assertEqual(float(saved["phase"]), 0.0)
        self.assertEqual(float(saved["frequency"]), 0.23)
        self.assertEqual(float(saved["amplitude"]), 0.7)
        self.assertEqual(int(saved["seed"]), 1006)
        self.assertEqual(int(saved["failure_step"]), 211)
        self.assertTrue(bool(saved["victim_failed"]))

    def test_non_failure_cannot_be_saved(self):
        trial = SineTrial(
            amplitude=0.5,
            failed=False,
            failure_step=500,
            trace=make_sine_trace(0.5),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                save_failure_trace(
                    os.path.join(temp_dir, "not-a-failure.npz"),
                    trial,
                    frequency=0.23,
                    seed=1006,
                    horizon=500,
                    max_wind=1.0,
                    victim_checkpoint="checkpoints/victim",
                    victim_env="cartpole",
                )

    def test_additional_seeds_exclude_discovery_seed(self):
        seeds = additional_seeds(1004, 5, excluded_seed=1006)
        self.assertEqual(seeds, [1004, 1005, 1007, 1008, 1009])


if __name__ == "__main__":
    unittest.main()
