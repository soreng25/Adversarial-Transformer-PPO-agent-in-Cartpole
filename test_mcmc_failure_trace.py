import unittest

import numpy as np

from mcmc_failure_trace import (
    acceptance_probability,
    is_in_bounds,
    load_source_trace,
    log_natural_density,
    propose_trace,
    run_mcmc,
)


class SourceTraceTests(unittest.TestCase):
    def test_episode_6_is_padded_to_350(self):
        source = load_source_trace(
            "adversary_wind_history_same_seeds.npz",
            episode_index=6,
            target_horizon=350,
            expected_sigma=1.0,
        )

        self.assertEqual(source.episode_length, 289)
        self.assertEqual(source.failure_step, 289)
        self.assertEqual(source.trace.shape, (350,))
        np.testing.assert_array_equal(source.trace[289:], np.zeros(61))
        self.assertEqual(source.max_wind, 1.0)
        self.assertEqual(source.natural_wind_sigma, 1.0)

    def test_zero_padding_does_not_change_log_score(self):
        source = load_source_trace(
            "adversary_wind_history_same_seeds.npz",
            episode_index=6,
            target_horizon=350,
            expected_sigma=1.0,
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


if __name__ == "__main__":
    unittest.main()
