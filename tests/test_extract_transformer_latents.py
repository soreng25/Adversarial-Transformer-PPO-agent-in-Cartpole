import os
import tempfile
import unittest

import numpy as np

from extract_transformer_latents import (
    advance_attention_memory,
    build_chain_layout,
    content_fingerprint,
    current_model_latent,
    evenly_spaced_indices,
    initialize_output_arrays,
    load_resume_arrays,
    output_payload,
    replay_and_summarize,
    save_output,
    summarize_latent_steps,
    validate_output_arrays,
    validate_rejected_rows,
)


class ChainLayoutTests(unittest.TestCase):
    def test_rejected_rows_map_to_the_previous_distinct_state(self):
        accepted = np.array([False, True, False, True], dtype=bool)

        unique_rows, state_ids, weights = build_chain_layout(accepted)

        np.testing.assert_array_equal(unique_rows, [0, 2, 4])
        np.testing.assert_array_equal(state_ids, [0, 0, 1, 1, 2])
        np.testing.assert_array_equal(weights, [2, 2, 1])
        self.assertEqual(int(np.sum(weights)), len(accepted) + 1)

    def test_all_rejections_produce_one_distinct_state(self):
        unique_rows, state_ids, weights = build_chain_layout(
            np.zeros(3, dtype=bool)
        )

        np.testing.assert_array_equal(unique_rows, [0])
        np.testing.assert_array_equal(state_ids, [0, 0, 0, 0])
        np.testing.assert_array_equal(weights, [4])

    def test_rejected_rows_must_repeat_trace_and_failure_step(self):
        chain = np.array(
            [
                [1.0, 2.0],
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=np.float32,
        )
        failure_steps = np.array([2, 2, 1], dtype=np.int32)
        accepted = np.array([False, True], dtype=bool)

        validate_rejected_rows(
            chain,
            failure_steps,
            accepted,
            chunk_size=1,
        )

        bad_chain = chain.copy()
        bad_chain[1, 0] = 9.0
        with self.assertRaisesRegex(ValueError, "rejected proposal"):
            validate_rejected_rows(
                bad_chain,
                failure_steps,
                accepted,
            )

        bad_steps = failure_steps.copy()
        bad_steps[1] = 1
        with self.assertRaisesRegex(ValueError, "rejected proposal"):
            validate_rejected_rows(chain, bad_steps, accepted)


class SelectionTests(unittest.TestCase):
    def test_validation_indices_span_the_available_states(self):
        np.testing.assert_array_equal(
            evenly_spaced_indices(length=5, count=3),
            [0, 2, 4],
        )

    def test_validation_count_is_capped_at_available_states(self):
        np.testing.assert_array_equal(
            evenly_spaced_indices(length=3, count=20),
            [0, 1, 2],
        )


class LatentSummaryTests(unittest.TestCase):
    def test_summaries_preserve_each_latent_coordinate(self):
        steps = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=np.float32,
        )

        mean, final_window_mean, final = summarize_latent_steps(
            steps,
            final_window=2,
        )

        np.testing.assert_array_equal(mean, [3.0, 4.0])
        np.testing.assert_array_equal(final_window_mean, [4.0, 5.0])
        np.testing.assert_array_equal(final, [5.0, 6.0])

    def test_large_final_window_uses_the_complete_trace(self):
        steps = np.array([[1.0], [3.0]], dtype=np.float32)

        mean, final_window_mean, _ = summarize_latent_steps(
            steps,
            final_window=50,
        )

        np.testing.assert_array_equal(mean, [2.0])
        np.testing.assert_array_equal(final_window_mean, [2.0])


class AttentionMemoryTests(unittest.TestCase):
    def test_new_state_is_appended_and_oldest_state_is_discarded(self):
        memory = [
            np.array(
                [
                    [1.0, 10.0],
                    [2.0, 20.0],
                    [3.0, 30.0],
                ],
                dtype=np.float32,
            )
        ]
        state_out = [np.array([4.0, 40.0], dtype=np.float32)]

        updated = advance_attention_memory(
            memory,
            state_out,
            latent_dim=2,
        )

        np.testing.assert_array_equal(
            updated[0],
            [
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
            ],
        )

    def test_multiple_state_output_vectors_are_rejected(self):
        memory = [np.zeros((3, 2), dtype=np.float32)]
        state_out = [np.zeros((2, 2), dtype=np.float32)]

        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            advance_attention_memory(memory, state_out, latent_dim=2)


class FakeModel:
    def __init__(self):
        self._features = None


class FakeAttentionPolicy:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.model = FakeModel()
        self.received_memory = []

    def get_initial_state(self):
        return [np.zeros(self.latent_dim, dtype=np.float32)]

    def compute_single_action(self, observation, state, explore, timestep):
        self.received_memory.append(np.asarray(state[0]).copy())
        value = float(np.asarray(observation).reshape(-1)[0])
        self.model._features = np.array(
            [[value, value + 0.5]],
            dtype=np.float32,
        )
        state_out = [
            np.full(self.latent_dim, 10.0 + value, dtype=np.float32)
        ]
        return np.array([0.0], dtype=np.float32), state_out, {}


class FakeFailureEnvironment:
    def __init__(self, failure_step):
        self.failure_step = failure_step
        self.steps = 0
        self.applied_winds = []

    def reset(self, seed):
        self.steps = 0
        self.applied_winds = []
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        self.applied_winds.append(float(np.asarray(action).reshape(-1)[0]))
        self.steps += 1
        failed = self.steps == self.failure_step
        info = {
            "victim_failed": failed,
            "episode_len": self.steps,
        }
        observation = np.array([float(self.steps)], dtype=np.float32)
        return observation, 0.0, failed, False, info


class ReplayTimingTests(unittest.TestCase):
    def test_latents_are_captured_before_each_forced_wind(self):
        policy = FakeAttentionPolicy(latent_dim=2)
        env = FakeFailureEnvironment(failure_step=3)

        summary = replay_and_summarize(
            env=env,
            adversary_policy=policy,
            trace=np.array([0.1, 0.2, 0.3, 9.9], dtype=np.float32),
            expected_failure_step=3,
            env_seed=7,
            latent_dim=2,
            memory_length=3,
            final_window=2,
        )

        # Observations before the three winds are 0, 1, and 2.
        np.testing.assert_array_equal(summary.mean, [1.0, 1.5])
        np.testing.assert_array_equal(
            summary.final_window_mean,
            [1.5, 2.0],
        )
        np.testing.assert_array_equal(summary.final, [2.0, 2.5])
        self.assertEqual(summary.step_count, 3)
        self.assertEqual(summary.replayed_failure_step, 3)
        np.testing.assert_allclose(env.applied_winds, [0.1, 0.2, 0.3])

        # The policy receives an unbatched rolling (memory, latent_dim) array.
        np.testing.assert_array_equal(
            policy.received_memory[0],
            np.zeros((3, 2), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            policy.received_memory[1],
            [[0.0, 0.0], [0.0, 0.0], [10.0, 10.0]],
        )
        np.testing.assert_array_equal(
            policy.received_memory[2],
            [[0.0, 0.0], [10.0, 10.0], [11.0, 11.0]],
        )


class ExtractionSafetyTests(unittest.TestCase):
    def test_multiple_model_feature_vectors_are_rejected(self):
        policy = FakeAttentionPolicy(latent_dim=2)
        policy.model._features = np.zeros((2, 2), dtype=np.float32)

        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            current_model_latent(policy, latent_dim=2)

    def test_content_fingerprint_changes_with_file_contents(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "data.bin")
            with open(path, "wb") as output_file:
                output_file.write(b"first")
            first = content_fingerprint(path)

            with open(path, "wb") as output_file:
                output_file.write(b"second")
            second = content_fingerprint(path)

        self.assertNotEqual(first, second)

    def test_resume_checks_identity_failure_steps_and_array_consistency(self):
        unique_rows = np.array([0, 2], dtype=np.int32)
        state_ids = np.array([0, 0, 1], dtype=np.int32)
        weights = np.array([2, 1], dtype=np.int32)
        failure_steps = np.array([3, 4], dtype=np.int32)
        identity = {
            "mcmc_path": "/data/chain.npz",
            "mcmc_file_sha256": "mcmc-hash",
            "victim_policy_sha256": "victim-hash",
            "adversary_policy_sha256": "adversary-hash",
            "env_seed": 7,
        }
        arrays = initialize_output_arrays(2, latent_dim=2)
        arrays["mean_latents"][0] = [1.0, 2.0]
        arrays["final_window_mean_latents"][0] = [1.5, 2.5]
        arrays["final_latents"][0] = [2.0, 3.0]
        arrays["latent_step_counts"][0] = 3
        arrays["replayed_failure_steps"][0] = 3
        arrays["completed"][0] = True

        validate_output_arrays(arrays, failure_steps, latent_dim=2)
        payload = output_payload(
            unique_chain_rows=unique_rows,
            chain_state_ids=state_ids,
            chain_weights=weights,
            unique_failure_steps=failure_steps,
            final_window=50,
            latent_dim=2,
            memory_length=3,
            run_identity=identity,
            **arrays
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "latents.npz")
            save_output(path, payload)
            self.assertTrue(os.path.isfile(path))
            self.assertFalse(os.path.exists(path + ".tmp.npz"))

            loaded = load_resume_arrays(
                path,
                unique_rows,
                state_ids,
                weights,
                failure_steps,
                latent_dim=2,
                memory_length=3,
                final_window=50,
                run_identity=identity,
            )
            np.testing.assert_array_equal(
                loaded["completed"],
                [True, False],
            )

            changed_identity = dict(identity)
            changed_identity["mcmc_file_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "mcmc file sha256"):
                load_resume_arrays(
                    path,
                    unique_rows,
                    state_ids,
                    weights,
                    failure_steps,
                    latent_dim=2,
                    memory_length=3,
                    final_window=50,
                    run_identity=changed_identity,
                )

            with self.assertRaisesRegex(ValueError, "failure steps"):
                load_resume_arrays(
                    path,
                    unique_rows,
                    state_ids,
                    weights,
                    np.array([3, 5], dtype=np.int32),
                    latent_dim=2,
                    memory_length=3,
                    final_window=50,
                    run_identity=identity,
                )

    def test_completed_rows_must_be_finite_and_match_failure_steps(self):
        arrays = initialize_output_arrays(1, latent_dim=2)
        arrays["completed"][0] = True
        arrays["latent_step_counts"][0] = 3
        arrays["replayed_failure_steps"][0] = 3

        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_output_arrays(
                arrays,
                np.array([3], dtype=np.int32),
                latent_dim=2,
                require_complete=True,
            )


if __name__ == "__main__":
    unittest.main()
