"""Cluster and visualize final pre-failure Transformer latent vectors."""

import argparse
import csv
import hashlib
import math
import os
import warnings
from dataclasses import dataclass

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


DEFAULT_LATENTS = "transformer_latent_summaries.npz"
DEFAULT_MCMC = "500000iteration_episode6_mcmc.npz"
DEFAULT_OUTPUT_DIR = "latent_analysis"
FINGERPRINT_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class AnalysisData:
    """Validated arrays linking distinct latent states to MCMC wind traces."""

    final_latents: np.ndarray
    unique_chain_rows: np.ndarray
    failure_steps: np.ndarray
    traces: np.ndarray


def content_fingerprint(path):
    """Return the extraction-compatible SHA-256 for a file or directory."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    digest = hashlib.sha256()
    if os.path.isfile(path):
        relative_files = [("", path)]
    else:
        relative_files = []
        for root, directories, files in os.walk(path):
            directories.sort()
            for filename in sorted(files):
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, path).replace(
                    os.sep,
                    "/",
                )
                relative_files.append((relative_path, full_path))

    for relative_path, full_path in relative_files:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with open(full_path, "rb") as input_file:
            while True:
                chunk = input_file.read(FINGERPRINT_CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def local_checkpoint_path(recorded_path):
    """Map a checkpoint path recorded on another host into this repository."""
    normalized = str(recorded_path).replace("\\", "/")
    marker = "checkpoints/"
    marker_position = normalized.find(marker)
    if marker_position < 0:
        raise ValueError(
            f"recorded checkpoint path does not contain {marker!r}: "
            f"{recorded_path}"
        )
    relative = normalized[marker_position:].split("/")
    return os.path.join(*relative)


def _require_keys(data, required, description):
    missing = sorted(set(required).difference(data.files))
    if missing:
        raise ValueError(f"{description} is missing arrays: {missing}")


def load_analysis_data(latent_path, mcmc_path, verify_checkpoints=True):
    """Load latent summaries and prove that they match the supplied MCMC data."""
    latent_required = {
        "unique_chain_rows",
        "failure_steps",
        "final_latents",
        "latent_step_counts",
        "replayed_failure_steps",
        "completed",
        "latent_dim",
        "mcmc_file_sha256",
        "victim_policy_checkpoint",
        "victim_policy_sha256",
        "adversary_policy_checkpoint",
        "adversary_policy_sha256",
    }
    with np.load(latent_path, allow_pickle=False) as latent_data:
        _require_keys(latent_data, latent_required, "latent result")
        saved = {
            key: np.array(latent_data[key], copy=True)
            for key in latent_required
        }

    with np.load(mcmc_path, allow_pickle=False) as mcmc:
        _require_keys(mcmc, {"chain", "failure_steps"}, "MCMC data")
        chain = np.asarray(mcmc["chain"], dtype=np.float32)
        all_failure_steps = np.asarray(mcmc["failure_steps"], dtype=np.int32)

    if content_fingerprint(mcmc_path) != str(saved["mcmc_file_sha256"]):
        raise ValueError("MCMC fingerprint does not match the latent extraction")

    if verify_checkpoints:
        checkpoint_fields = (
            ("victim_policy_checkpoint", "victim_policy_sha256"),
            ("adversary_policy_checkpoint", "adversary_policy_sha256"),
        )
        for path_field, hash_field in checkpoint_fields:
            checkpoint = local_checkpoint_path(str(saved[path_field]))
            if content_fingerprint(checkpoint) != str(saved[hash_field]):
                raise ValueError(
                    f"{path_field} fingerprint does not match the extraction"
                )

    unique_rows = np.asarray(saved["unique_chain_rows"], dtype=np.int32)
    failure_steps = np.asarray(saved["failure_steps"], dtype=np.int32)
    final_latents = np.asarray(saved["final_latents"], dtype=np.float32)
    completed = np.asarray(saved["completed"], dtype=bool)

    number_of_states = len(unique_rows)
    if unique_rows.shape != (number_of_states,) or number_of_states < 2:
        raise ValueError("unique_chain_rows must contain at least two states")
    if np.any(np.diff(unique_rows) <= 0):
        raise ValueError("unique_chain_rows must be strictly increasing")
    if unique_rows[0] < 0 or unique_rows[-1] >= len(chain):
        raise ValueError("unique_chain_rows contains an out-of-range MCMC row")
    if failure_steps.shape != (number_of_states,):
        raise ValueError("failure_steps does not match latent rows")
    if completed.shape != (number_of_states,) or not np.all(completed):
        raise ValueError("latent extraction is incomplete")

    latent_dim = int(saved["latent_dim"])
    if latent_dim != 64 or final_latents.shape != (number_of_states, latent_dim):
        raise ValueError(
            f"expected final_latents shape ({number_of_states}, 64), "
            f"got {final_latents.shape}"
        )
    if not np.all(np.isfinite(final_latents)):
        raise ValueError("final_latents contains non-finite values")

    if chain.shape[0] != all_failure_steps.shape[0]:
        raise ValueError("MCMC chain and failure_steps row counts differ")
    if not np.array_equal(failure_steps, all_failure_steps[unique_rows]):
        raise ValueError("latent failure_steps do not match the MCMC rows")
    if not np.array_equal(saved["latent_step_counts"], failure_steps):
        raise ValueError("latent step counts do not match failure steps")
    if not np.array_equal(saved["replayed_failure_steps"], failure_steps):
        raise ValueError("replayed failure steps do not match source failures")
    if np.any(failure_steps < 1) or np.any(failure_steps > chain.shape[1]):
        raise ValueError("failure_steps contains values outside the trace width")

    return AnalysisData(
        final_latents=final_latents,
        unique_chain_rows=unique_rows,
        failure_steps=failure_steps,
        traces=chain[unique_rows].copy(),
    )


def select_kmeans(
    standardized_latents,
    min_k=2,
    max_k=10,
    random_seed=42,
    silhouette_sample_size=5000,
):
    """Select K-means k by silhouette score, preferring smaller tied values."""
    values = np.asarray(standardized_latents, dtype=np.float64)
    if values.ndim != 2 or len(values) < 3:
        raise ValueError("standardized_latents must contain at least three rows")
    if min_k < 2 or max_k < min_k or max_k >= len(values):
        raise ValueError("cluster range must satisfy 2 <= min_k <= max_k < rows")

    scores = {}
    models = {}
    sample_size = min(int(silhouette_sample_size), len(values))
    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=random_seed)
        labels = model.fit_predict(values)
        score = silhouette_score(
            values,
            labels,
            sample_size=sample_size,
            random_state=random_seed,
        )
        scores[k] = float(score)
        models[k] = model

    best_k = min(scores, key=lambda k: (-scores[k], k))
    return models[best_k].labels_.astype(np.int32), best_k, scores


def remap_clusters_by_failure_time(labels, failure_steps):
    """Number clusters by increasing mean failure step."""
    labels = np.asarray(labels, dtype=np.int32)
    failure_steps = np.asarray(failure_steps, dtype=np.float64)
    if labels.shape != failure_steps.shape:
        raise ValueError("labels and failure_steps must have matching shapes")

    original_clusters = np.unique(labels)
    ordered = sorted(
        original_clusters,
        key=lambda cluster: (
            float(np.mean(failure_steps[labels == cluster])),
            int(cluster),
        ),
    )
    mapping = {old: new for new, old in enumerate(ordered)}
    remapped = np.asarray([mapping[int(label)] for label in labels], dtype=np.int32)
    return remapped


def compute_embeddings(
    standardized_latents,
    *,
    random_seed,
    tsne_perplexity,
    umap_neighbors,
    umap_min_dist,
):
    """Return PCA, t-SNE, and UMAP coordinates for the same latent rows."""
    import umap

    number_of_rows = len(standardized_latents)
    if tsne_perplexity >= number_of_rows:
        raise ValueError("t-SNE perplexity must be smaller than the row count")
    if umap_neighbors >= number_of_rows:
        raise ValueError("UMAP neighbors must be smaller than the row count")

    pca_model = PCA(n_components=2)
    pca_coordinates = pca_model.fit_transform(standardized_latents)
    tsne_coordinates = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=1000,
        random_state=random_seed,
    ).fit_transform(standardized_latents)
    umap_coordinates = umap.UMAP(
        n_components=2,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        metric="euclidean",
        random_state=random_seed,
        n_jobs=1,
    ).fit_transform(standardized_latents)

    return {
        "pca": np.asarray(pca_coordinates, dtype=np.float32),
        "tsne": np.asarray(tsne_coordinates, dtype=np.float32),
        "umap": np.asarray(umap_coordinates, dtype=np.float32),
        "pca_explained_variance": pca_model.explained_variance_ratio_.copy(),
    }


def cluster_colors(number_of_clusters):
    cmap = plt.get_cmap("tab10")
    return [cmap(index % 10) for index in range(number_of_clusters)]


def save_cluster_selection_plot(scores, output_path):
    ordered_k = sorted(scores)
    plt.figure(figsize=(8, 5))
    plt.plot(
        ordered_k,
        [scores[k] for k in ordered_k],
        marker="o",
        color="black",
    )
    plt.xlabel("Number of K-means Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Latent-Space Cluster Selection")
    plt.xticks(ordered_k)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_embedding_plot(
    coordinates,
    labels,
    output_path,
    title,
    x_label,
    y_label,
):
    number_of_clusters = int(np.max(labels)) + 1
    colors = cluster_colors(number_of_clusters)
    plt.figure(figsize=(10, 8))
    for cluster in range(number_of_clusters):
        selected = labels == cluster
        plt.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=8,
            alpha=0.55,
            color=colors[cluster],
            label=f"Cluster {cluster}",
            rasterized=True,
        )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(markerscale=2)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def cluster_mean_traces(traces, failure_steps, labels):
    """Compute per-cluster means using only winds applied before failure."""
    traces = np.asarray(traces, dtype=np.float64)
    failure_steps = np.asarray(failure_steps, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32)
    if traces.ndim != 2 or len(traces) != len(failure_steps):
        raise ValueError("traces and failure_steps do not align")
    if labels.shape != failure_steps.shape:
        raise ValueError("labels and failure_steps do not align")

    active = np.arange(traces.shape[1])[None, :] < failure_steps[:, None]
    means = {}
    for cluster in np.unique(labels):
        selected = labels == cluster
        cluster_active = active[selected]
        cluster_values = np.where(cluster_active, traces[selected], 0.0)
        counts = np.sum(cluster_active, axis=0)
        mean_trace = np.full(traces.shape[1], np.nan, dtype=np.float64)
        populated = counts > 0
        mean_trace[populated] = (
            np.sum(cluster_values[:, populated], axis=0) / counts[populated]
        )
        means[int(cluster)] = mean_trace
    return means


def _cluster_legend(labels, colors):
    return [
        Line2D(
            [0],
            [0],
            color=colors[cluster],
            linewidth=2,
            label=f"Cluster {cluster}",
        )
        for cluster in range(int(np.max(labels)) + 1)
    ]


def save_all_trace_plot(traces, failure_steps, labels, output_path):
    number_of_clusters = int(np.max(labels)) + 1
    colors = cluster_colors(number_of_clusters)
    means = cluster_mean_traces(traces, failure_steps, labels)

    plt.figure(figsize=(16, 8))
    for index, (trace, failure_step) in enumerate(zip(traces, failure_steps)):
        cluster = int(labels[index])
        plt.plot(
            np.arange(failure_step),
            trace[:failure_step],
            color=colors[cluster],
            linewidth=0.35,
            alpha=0.025,
            rasterized=True,
        )
    for cluster, mean_trace in means.items():
        valid = np.isfinite(mean_trace)
        plt.plot(
            np.flatnonzero(valid),
            mean_trace[valid],
            color=colors[cluster],
            linewidth=2.2,
        )
    plt.xlabel("Timestep")
    plt.ylabel("Applied Wind")
    plt.title("All Distinct Failure Traces Colored by Final-Latent Cluster")
    plt.legend(handles=_cluster_legend(labels, colors))
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_faceted_trace_plot(traces, failure_steps, labels, output_path):
    number_of_clusters = int(np.max(labels)) + 1
    colors = cluster_colors(number_of_clusters)
    means = cluster_mean_traces(traces, failure_steps, labels)
    columns = min(3, number_of_clusters)
    rows = math.ceil(number_of_clusters / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(6 * columns, 4 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for cluster, ax in enumerate(axes.flat):
        if cluster >= number_of_clusters:
            ax.set_visible(False)
            continue
        selected_indices = np.flatnonzero(labels == cluster)
        for index in selected_indices:
            failure_step = int(failure_steps[index])
            ax.plot(
                np.arange(failure_step),
                traces[index, :failure_step],
                color=colors[cluster],
                linewidth=0.4,
                alpha=0.035,
                rasterized=True,
            )
        mean_trace = means[cluster]
        valid = np.isfinite(mean_trace)
        ax.plot(
            np.flatnonzero(valid),
            mean_trace[valid],
            color=colors[cluster],
            linewidth=2.2,
            label="Cluster Mean",
        )
        ax.set_title(f"Cluster {cluster}: {len(selected_indices):,} traces")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")

    fig.supxlabel("Timestep")
    fig.supylabel("Applied Wind")
    fig.suptitle("Distinct Failure Traces Separated by Final-Latent Cluster")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_cluster_summaries(labels, failure_steps):
    """Return CSV-ready statistics treating every distinct trace once."""
    labels = np.asarray(labels, dtype=np.int32)
    failure_steps = np.asarray(failure_steps, dtype=np.float64)
    if labels.shape != failure_steps.shape:
        raise ValueError("labels and failure_steps must have matching shapes")

    total_traces = len(labels)
    summaries = []
    for cluster in np.unique(labels):
        selected = labels == cluster
        selected_steps = failure_steps[selected]
        trace_count = int(np.sum(selected))
        summaries.append(
            {
                "cluster": int(cluster),
                "trace_count": trace_count,
                "trace_prevalence": trace_count / total_traces,
                "failure_step_min": int(np.min(selected_steps)),
                "failure_step_25th": float(np.percentile(selected_steps, 25)),
                "failure_step_median": float(np.median(selected_steps)),
                "failure_step_mean": float(np.mean(selected_steps)),
                "failure_step_75th": float(np.percentile(selected_steps, 75)),
                "failure_step_max": int(np.max(selected_steps)),
            }
        )
    return summaries


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_result_tables(data, labels, embeddings, output_dir):
    state_rows = []
    for state_id in range(len(labels)):
        state_rows.append(
            {
                "unique_state_id": state_id,
                "mcmc_row": int(data.unique_chain_rows[state_id]),
                "failure_step": int(data.failure_steps[state_id]),
                "cluster": int(labels[state_id]),
                "pca_1": float(embeddings["pca"][state_id, 0]),
                "pca_2": float(embeddings["pca"][state_id, 1]),
                "tsne_1": float(embeddings["tsne"][state_id, 0]),
                "tsne_2": float(embeddings["tsne"][state_id, 1]),
                "umap_1": float(embeddings["umap"][state_id, 0]),
                "umap_2": float(embeddings["umap"][state_id, 1]),
            }
        )
    state_fields = list(state_rows[0])
    write_csv(
        os.path.join(output_dir, "latent_clusters.csv"),
        state_rows,
        state_fields,
    )

    summaries = build_cluster_summaries(
        labels,
        data.failure_steps,
    )
    write_csv(
        os.path.join(output_dir, "cluster_summary.csv"),
        summaries,
        list(summaries[0]),
    )
    return summaries


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster and visualize final pre-failure Transformer latents."
    )
    parser.add_argument("--latents", default=DEFAULT_LATENTS)
    parser.add_argument("--mcmc", default=DEFAULT_MCMC)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    return parser.parse_args()


def validate_args(args):
    if args.min_k < 2 or args.max_k < args.min_k:
        raise ValueError("cluster range must satisfy 2 <= min-k <= max-k")
    if args.tsne_perplexity <= 0:
        raise ValueError("--tsne-perplexity must be positive")
    if args.umap_neighbors < 2:
        raise ValueError("--umap-neighbors must be at least 2")
    if args.umap_min_dist < 0:
        raise ValueError("--umap-min-dist cannot be negative")


def main():
    args = parse_args()
    validate_args(args)

    print("Loading and validating latent and MCMC data...", flush=True)
    data = load_analysis_data(args.latents, args.mcmc)
    if args.max_k >= len(data.final_latents):
        raise ValueError("--max-k must be smaller than the number of states")

    print(
        f"Loaded {len(data.final_latents):,} distinct 64D final latents. "
        "Each trace will be analyzed once."
    )
    standardized = StandardScaler().fit_transform(data.final_latents)

    print(
        f"Selecting K-means cluster count from {args.min_k} to {args.max_k}...",
        flush=True,
    )
    raw_labels, best_k, silhouette_scores = select_kmeans(
        standardized,
        min_k=args.min_k,
        max_k=args.max_k,
        random_seed=args.random_seed,
    )
    labels = remap_clusters_by_failure_time(raw_labels, data.failure_steps)

    print("Computing PCA, t-SNE, and UMAP embeddings...", flush=True)
    embeddings = compute_embeddings(
        standardized,
        random_seed=args.random_seed,
        tsne_perplexity=args.tsne_perplexity,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    save_cluster_selection_plot(
        silhouette_scores,
        os.path.join(args.output_dir, "cluster_selection.png"),
    )
    explained = embeddings["pca_explained_variance"]
    save_embedding_plot(
        embeddings["pca"],
        labels,
        os.path.join(args.output_dir, "pca_clusters.png"),
        "PCA of Final Pre-Failure Transformer Latents",
        f"PC1 ({100 * explained[0]:.1f}% variance)",
        f"PC2 ({100 * explained[1]:.1f}% variance)",
    )
    save_embedding_plot(
        embeddings["tsne"],
        labels,
        os.path.join(args.output_dir, "tsne_clusters.png"),
        "t-SNE of Final Pre-Failure Transformer Latents",
        "t-SNE 1",
        "t-SNE 2",
    )
    save_embedding_plot(
        embeddings["umap"],
        labels,
        os.path.join(args.output_dir, "umap_clusters.png"),
        "UMAP of Final Pre-Failure Transformer Latents",
        "UMAP 1",
        "UMAP 2",
    )
    save_all_trace_plot(
        data.traces,
        data.failure_steps,
        labels,
        os.path.join(args.output_dir, "wind_traces_all_clusters.png"),
    )
    save_faceted_trace_plot(
        data.traces,
        data.failure_steps,
        labels,
        os.path.join(args.output_dir, "wind_traces_by_cluster.png"),
    )
    summaries = save_result_tables(data, labels, embeddings, args.output_dir)

    print()
    print(f"selected_clusters={best_k}")
    print(
        "pca_explained_variance="
        f"{100 * explained[0]:.2f}% + {100 * explained[1]:.2f}%"
    )
    print("cluster summaries:")
    for summary in summaries:
        print(
            "  cluster={cluster} traces={trace_count} prevalence="
            "{trace_prevalence:.3f} mean_failure_step="
            "{failure_step_mean:.1f}".format(**summary)
        )
    print(f"outputs={os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
