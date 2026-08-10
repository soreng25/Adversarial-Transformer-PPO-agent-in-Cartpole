import os
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import numpy as np

from analyze_transformer_latents import (
    build_cluster_summaries,
    cluster_mean_traces,
    content_fingerprint,
    load_analysis_data,
    remap_clusters_by_failure_time,
    save_all_trace_plot,
    save_cluster_selection_plot,
    save_embedding_plot,
    save_faceted_trace_plot,
    select_kmeans,
)


def write_synthetic_inputs(directory, *, incomplete=False, nonfinite=False):
    mcmc_path = os.path.join(directory, "mcmc.npz")
    chain = np.arange(24, dtype=np.float32).reshape(6, 4) / 24
    all_failure_steps = np.array([2, 2, 3, 3, 4, 4], dtype=np.int32)
    np.savez(mcmc_path, chain=chain, failure_steps=all_failure_steps)

    unique_rows = np.array([0, 2, 4], dtype=np.int32)
    latents = np.zeros((3, 64), dtype=np.float32)
    latents[1] = 1
    latents[2] = 2
    if nonfinite:
        latents[1, 3] = np.nan

    latent_path = os.path.join(directory, "latents.npz")
    np.savez(
        latent_path,
        unique_chain_rows=unique_rows,
        failure_steps=all_failure_steps[unique_rows],
        final_latents=latents,
        latent_step_counts=all_failure_steps[unique_rows],
        replayed_failure_steps=all_failure_steps[unique_rows],
        completed=np.array([True, not incomplete, True]),
        latent_dim=np.int32(64),
        mcmc_file_sha256=np.str_(content_fingerprint(mcmc_path)),
        victim_policy_checkpoint=np.str_("checkpoints/victim"),
        victim_policy_sha256=np.str_("unused"),
        adversary_policy_checkpoint=np.str_("checkpoints/adversary"),
        adversary_policy_sha256=np.str_("unused"),
    )
    return latent_path, mcmc_path


class InputValidationTests(unittest.TestCase):
    def test_valid_inputs_link_unique_states_to_mcmc(self):
        with tempfile.TemporaryDirectory() as directory:
            latent_path, mcmc_path = write_synthetic_inputs(directory)
            data = load_analysis_data(
                latent_path,
                mcmc_path,
                verify_checkpoints=False,
            )

        self.assertEqual(data.final_latents.shape, (3, 64))
        np.testing.assert_array_equal(data.unique_chain_rows, [0, 2, 4])
        np.testing.assert_array_equal(data.failure_steps, [2, 3, 4])
        np.testing.assert_allclose(data.traces[:, 0], [0.0, 8 / 24, 16 / 24])

    def test_incomplete_and_nonfinite_results_are_rejected(self):
        for keyword in ("incomplete", "nonfinite"):
            with self.subTest(keyword=keyword):
                with tempfile.TemporaryDirectory() as directory:
                    paths = write_synthetic_inputs(
                        directory,
                        **{keyword: True},
                    )
                    with self.assertRaises(ValueError):
                        load_analysis_data(*paths, verify_checkpoints=False)

    def test_mismatched_failure_steps_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            latent_path, mcmc_path = write_synthetic_inputs(directory)
            with np.load(mcmc_path) as source:
                chain = source["chain"].copy()
                steps = source["failure_steps"].copy()
            steps[2] = 4
            np.savez(mcmc_path, chain=chain, failure_steps=steps)
            with self.assertRaises(ValueError):
                load_analysis_data(
                    latent_path,
                    mcmc_path,
                    verify_checkpoints=False,
                )


class ClusteringTests(unittest.TestCase):
    def test_cluster_selection_is_reproducible(self):
        rng = np.random.default_rng(4)
        values = np.vstack(
            [
                rng.normal(-5, 0.1, size=(30, 4)),
                rng.normal(0, 0.1, size=(30, 4)),
                rng.normal(5, 0.1, size=(30, 4)),
            ]
        )

        first = select_kmeans(values, 2, 4, random_seed=42)
        second = select_kmeans(values, 2, 4, random_seed=42)

        self.assertEqual(first[1], 3)
        self.assertEqual(first[1], second[1])
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[2], second[2])

    def test_cluster_labels_are_ordered_by_failure_time(self):
        labels = np.array([7, 7, 2, 2, 9, 9])
        failure_steps = np.array([200, 220, 100, 120, 150, 160])
        remapped = remap_clusters_by_failure_time(labels, failure_steps)
        np.testing.assert_array_equal(remapped, [2, 2, 0, 0, 1, 1])

    def test_cluster_summary_counts_each_trace_once(self):
        summaries = build_cluster_summaries(
            labels=np.array([0, 0, 1]),
            failure_steps=np.array([100, 200, 300]),
        )
        self.assertEqual(summaries[0]["trace_count"], 2)
        self.assertEqual(summaries[1]["trace_count"], 1)
        self.assertAlmostEqual(
            sum(row["trace_prevalence"] for row in summaries),
            1.0,
        )


class TraceAndPlotTests(unittest.TestCase):
    def test_cluster_means_exclude_post_failure_values(self):
        traces = np.array(
            [
                [1.0, 2.0, 100.0, 100.0],
                [3.0, 4.0, 5.0, 100.0],
            ]
        )
        means = cluster_mean_traces(
            traces,
            failure_steps=np.array([2, 3]),
            labels=np.array([0, 0]),
        )
        np.testing.assert_allclose(means[0][:3], [2.0, 3.0, 5.0])
        self.assertTrue(np.isnan(means[0][3]))

    def test_all_plot_types_render_from_synthetic_data(self):
        rng = np.random.default_rng(3)
        coordinates = rng.normal(size=(6, 2))
        labels = np.array([0, 0, 0, 1, 1, 1])
        traces = rng.normal(size=(6, 5))
        failure_steps = np.array([2, 3, 4, 3, 4, 5])

        with tempfile.TemporaryDirectory() as directory:
            paths = [
                os.path.join(directory, "selection.png"),
                os.path.join(directory, "pca.png"),
                os.path.join(directory, "tsne.png"),
                os.path.join(directory, "umap.png"),
                os.path.join(directory, "all.png"),
                os.path.join(directory, "facets.png"),
            ]
            save_cluster_selection_plot({2: 0.5, 3: 0.4}, paths[0])
            for path, title in zip(paths[1:4], ("PCA", "t-SNE", "UMAP")):
                save_embedding_plot(
                    coordinates,
                    labels,
                    path,
                    title,
                    "x",
                    "y",
                )
            save_all_trace_plot(traces, failure_steps, labels, paths[4])
            save_faceted_trace_plot(traces, failure_steps, labels, paths[5])
            self.assertTrue(all(os.path.getsize(path) > 0 for path in paths))


if __name__ == "__main__":
    unittest.main()
