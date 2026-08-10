import csv
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from plot_wind_history import load_csv_history, load_history, plot_history


class CsvHistoryTests(unittest.TestCase):
    def write_csv(self, path):
        with open(path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "trace_id",
                    "chain_iteration",
                    "failure_step",
                    "timestep",
                    "wind",
                ]
            )
            writer.writerows(
                [
                    [0, 0, 2, 1, 0.1],
                    [0, 0, 2, 2, 0.2],
                    [1, 4, 3, 1, -0.1],
                    [1, 4, 3, 2, -0.2],
                    [1, 4, 3, 3, -0.3],
                ]
            )

    def test_load_csv_builds_nan_padded_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "chain.csv")
            self.write_csv(path)
            history = load_csv_history(path)

        self.assertEqual(history["input_format"], "csv")
        self.assertEqual(history["winds"].shape, (2, 3))
        np.testing.assert_allclose(history["winds"][0, :2], [0.1, 0.2])
        self.assertTrue(np.isnan(history["winds"][0, 2]))
        np.testing.assert_array_equal(history["failure_steps"], [2, 3])
        np.testing.assert_array_equal(history["chain_iterations"], [0, 4])
        np.testing.assert_array_equal(history["timesteps"], [1, 2, 3])

    def test_load_history_dispatches_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "chain.csv")
            self.write_csv(path)
            history = load_history(path)

        self.assertEqual(history["input_format"], "csv")

    def test_csv_history_renders_png(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "chain.csv")
            png_path = os.path.join(temp_dir, "chain.png")
            self.write_csv(csv_path)
            history = load_csv_history(csv_path)
            args = SimpleNamespace(
                burn_in=0,
                thin=1,
                failures_only=False,
                episodes=None,
                line_alpha=0.65,
                legend_episodes=20,
                show_mean=True,
                show_std=False,
                show_failures=True,
                out_path=png_path,
                dpi=72,
            )

            plot_history(history, args)

            self.assertTrue(os.path.isfile(png_path))
            self.assertGreater(os.path.getsize(png_path), 0)

    def test_csv_rejects_missing_timestep(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "chain.csv")
            self.write_csv(path)
            with open(path, "r", newline="") as csv_file:
                rows = list(csv.reader(csv_file))
            with open(path, "w", newline="") as csv_file:
                csv.writer(csv_file).writerows(rows[:-1])

            with self.assertRaisesRegex(ValueError, "every timestep"):
                load_csv_history(path)


if __name__ == "__main__":
    unittest.main()
