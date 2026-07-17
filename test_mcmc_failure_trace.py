import csv
import os
import tempfile
import unittest

import numpy as np

from mcmc_failure_trace import (
    acceptance_probability,
    is_in_bounds,
    load_source_trace,
    log_natural_density,
    propose_trace,
    run_mcmc,
    save_chain_csv,
)


class SourceTraceTests(unittest.TestCase):
    def test_episode_6_is_padded_to_350(self):
        source = load_source_trace(
            "adversary_wind_history_same_seeds.npz",
            episode_index=6,
            target_horizon=350,
        )

        self.assertEqual(source.episode_length, 289)
        self.assertEqual(source.failure_step, 289)
        self.assertEqual(source.trace.shape, (350,))
        np.testing.assert_array_equal(source.trace[289:], np.zeros(61))
        self.assertEqual(source.max_wind, 1.0)
        self.assertEqual(source.source_wind_sigma, 1.0)

    def test_zero_padding_does_not_change_log_score(self):
        source = load_source_trace(
            "adversary_wind_history_same_seeds.npz",
            episode_index=6,
            target_horizon=350,
        )

        original_score = log_natural_density(
            source.trace[: source.episode_length],
            sigma=1.0,
        )
        padded_score = log_natural_density(source.trace, sigma=1.0)
        self.assertAlmostEqual(original_score, padded_score)


class MetropolisMathTests(unittest.TestCase):
    def test_acceptance_probability_for_four_to_one_ratio(self):
        self.assertEqual(acceptance_probability(0.0, np.log(4.0)), 1.0)
        self.assertAlmostEqual(
            acceptance_probability(np.log(4.0), 0.0),
            0.25,
        )
        self.assertEqual(acceptance_probability(0.0, -np.inf), 0.0)

    def test_gaussian_log_score(self):
        trace = np.array([1.0, -2.0, 0.0])
        self.assertAlmostEqual(log_natural_density(trace, sigma=1.0), -2.5)
        self.assertAlmostEqual(log_natural_density(trace, sigma=2.0), -0.625)

    def test_full_vector_proposal_uses_requested_sigma(self):
        rng = np.random.default_rng(7)
        current = np.zeros(100_000)
        proposed = propose_trace(current, rng, sigma=0.01, mode="all")

        self.assertAlmostEqual(float(np.mean(proposed)), 0.0, places=4)
        self.assertAlmostEqual(float(np.std(proposed)), 0.01, places=4)
        self.assertGreater(np.count_nonzero(proposed), 99_900)

    def test_single_and_block_proposals_change_expected_coordinates(self):
        current = np.zeros(350)
        single = propose_trace(
            current,
            np.random.default_rng(1),
            sigma=0.01,
            mode="single",
        )
        block = propose_trace(
            current,
            np.random.default_rng(2),
            sigma=0.01,
            mode="block",
            block_size=10,
        )

        self.assertEqual(np.count_nonzero(single), 1)
        self.assertEqual(np.count_nonzero(block), 10)

    def test_bounds_check(self):
        self.assertTrue(is_in_bounds(np.array([-1.0, 0.0, 1.0]), 1.0))
        self.assertFalse(is_in_bounds(np.array([0.0, 1.0001]), 1.0))


class ChainTests(unittest.TestCase):
    @staticmethod
    def replay(trace):
        # A small deterministic stand-in for the victim simulator.  Any trace
        # with a positive first coordinate fails at step 2.
        return bool(trace[0] > 0.0), 2

    def run_chain(self, seed):
        return run_mcmc(
            initial_trace=np.array([0.1, 0.0, 0.0]),
            replay=self.replay,
            iterations=100,
            proposal_sigma=0.2,
            natural_wind_sigma=1.0,
            max_wind=10.0,
            rng=np.random.default_rng(seed),
            proposal_mode="all",
        )

    def test_chain_is_reproducible_and_contains_only_failures(self):
        first = self.run_chain(seed=11)
        second = self.run_chain(seed=11)

        np.testing.assert_array_equal(first.chain, second.chain)
        np.testing.assert_array_equal(first.accepted, second.accepted)
        self.assertTrue(np.all(first.chain[:, 0] > 0.0))
        self.assertTrue(np.all(first.failure_steps == 2))

    def test_rejections_repeat_the_current_state(self):
        result = self.run_chain(seed=13)
        rejected = np.flatnonzero(~result.accepted)
        self.assertGreater(len(rejected), 0)
        for index in rejected:
            np.testing.assert_array_equal(
                result.chain[index + 1],
                result.chain[index],
            )

    def test_csv_contains_initial_and_accepted_applied_winds(self):
        result = self.run_chain(seed=13)
        expected_trace_count = 1 + int(np.sum(result.accepted))
        expected_data_rows = int(result.failure_steps[0])
        expected_data_rows += int(
            np.sum(result.failure_steps[1:][result.accepted])
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "chain.csv")
            trace_count = save_chain_csv(path, result)
            with open(path, newline="") as csv_file:
                rows = list(csv.reader(csv_file))

        self.assertEqual(trace_count, expected_trace_count)
        self.assertEqual(
            rows[0],
            [
                "trace_id",
                "chain_iteration",
                "failure_step",
                "timestep",
                "wind",
            ],
        )
        self.assertEqual(len(rows) - 1, expected_data_rows)


if __name__ == "__main__":
    unittest.main()
