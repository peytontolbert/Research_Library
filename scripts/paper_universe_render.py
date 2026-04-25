from __future__ import annotations

"""
Render lightweight 3D overview assets for the standalone paper universe.

Outputs written under the paper universe directory:

- universe_3d.png
- universe_3d_detailed.png
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq  # type: ignore


DEFAULT_UNIVERSE_DIR = Path("/data/repository_library/exports/_paper_universe")
DEFAULT_OVERVIEW_SAMPLE = 120_000
DEFAULT_DETAILED_SAMPLE = 40_000
DEFAULT_SEED = 42


def _read_table(path: Path, columns: list[str]):
    parquet_file = pq.ParquetFile(path)
    available = set(parquet_file.schema_arrow.names)
    selected = [column for column in columns if column in available]
    return parquet_file.read(columns=selected)


def _column_numpy(table, name: str, dtype=np.float32, default: float | int = 0) -> np.ndarray:
    if name not in table.column_names:
        return np.full(int(table.num_rows), default, dtype=dtype)
    return np.asarray(table.column(name).to_numpy(zero_copy_only=False), dtype=dtype)


def _column_strings(table, name: str, default: str = "") -> list[str]:
    if name not in table.column_names:
        return [default for _ in range(int(table.num_rows))]
    return [str(v or "") for v in table.column(name).to_pylist()]


def _sample_indices(total_rows: int, sample_size: int, seed: int) -> np.ndarray:
    if total_rows <= 0:
        return np.asarray([], dtype=np.int64)
    cap = max(1, int(sample_size or 0))
    if total_rows <= cap:
        return np.arange(total_rows, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(total_rows, size=cap, replace=False))


def _style_axes(ax, *, title: str) -> None:
    ax.set_title(title, fontsize=14, pad=16, color="#0f172a")
    ax.set_xlabel("X", color="#334155")
    ax.set_ylabel("Y", color="#334155")
    ax.set_zlabel("Z", color="#334155")
    ax.xaxis.pane.set_facecolor((0.96, 0.97, 0.99, 1.0))
    ax.yaxis.pane.set_facecolor((0.96, 0.97, 0.99, 1.0))
    ax.zaxis.pane.set_facecolor((0.96, 0.97, 0.99, 1.0))
    ax.grid(True, alpha=0.18)
    ax.view_init(elev=22, azim=34)


def _truncate(text: str, *, max_chars: int = 40) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max(0, max_chars - 3)].rstrip()}..."


def _render_overview(
    universe_dir: Path,
    *,
    paper_sample: int,
    seed: int,
    output_path: Path,
) -> Dict[str, int]:
    paper_nodes = _read_table(universe_dir / "paper_nodes.parquet", ["x", "y", "z", "primary_category", "year"])
    category_nodes = _read_table(universe_dir / "category_nodes.parquet", ["category_id", "paper_count"])
    year_nodes = _read_table(universe_dir / "year_nodes.parquet", ["year"])

    px = _column_numpy(paper_nodes, "x")
    py = _column_numpy(paper_nodes, "y")
    pz = _column_numpy(paper_nodes, "z")
    paper_categories = _column_strings(paper_nodes, "primary_category")
    paper_years = _column_numpy(paper_nodes, "year", dtype=np.int32)
    paper_idx = _sample_indices(len(px), paper_sample, seed)

    category_ids = _column_strings(category_nodes, "category_id")
    years = np.asarray(year_nodes.column("year").to_pylist(), dtype=np.int32)
    sampled_categories = [paper_categories[idx] for idx in paper_idx.tolist() if paper_categories[idx]]
    sampled_years = paper_years[paper_idx]
    top_sampled_categories = Counter(sampled_categories).most_common(5)

    fig = plt.figure(figsize=(14, 11), dpi=220, facecolor="#f8fafc")
    ax = fig.add_subplot(111, projection="3d")
    _style_axes(ax, title="Paper Universe 3D Overview")

    valid_sampled_years = sampled_years[sampled_years > 0]
    if len(valid_sampled_years) >= 2 and int(valid_sampled_years.min()) != int(valid_sampled_years.max()):
        scatter = ax.scatter(
            px[paper_idx],
            py[paper_idx],
            pz[paper_idx],
            s=0.8,
            alpha=0.22,
            c=sampled_years,
            cmap="viridis",
            linewidths=0,
            depthshade=False,
        )
        colorbar = fig.colorbar(scatter, ax=ax, shrink=0.66, pad=0.03)
        colorbar.set_label("Publication year", color="#334155")
    else:
        ax.scatter(
            px[paper_idx],
            py[paper_idx],
            pz[paper_idx],
            s=0.8,
            alpha=0.22,
            c="#1d4ed8",
            linewidths=0,
            depthshade=False,
        )

    top_categories_text = ", ".join(f"{category} ({count:,})" for category, count in top_sampled_categories) or "n/a"

    ax.text2D(
        0.02,
        0.98,
        (
            f"paper nodes shown: {len(paper_idx):,} / {len(px):,}\n"
            f"categories represented: {len(set(sampled_categories)):,} / {len(category_ids):,}\n"
            f"years represented: {len(set(int(year) for year in sampled_years.tolist() if int(year) > 0)):,} / {len(years):,}\n"
            f"top sampled categories: {top_categories_text}"
        ),
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        color="#334155",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.4"},
    )

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "paper_points": int(len(paper_idx)),
        "categories_represented": int(len(set(sampled_categories))),
        "years_represented": int(len(set(int(year) for year in sampled_years.tolist() if int(year) > 0))),
    }


def _render_detailed(
    universe_dir: Path,
    *,
    paper_sample: int,
    seed: int,
    output_path: Path,
) -> Dict[str, int]:
    paper_nodes = _read_table(universe_dir / "paper_nodes.parquet", ["x", "y", "z", "primary_category", "title"])
    category_nodes = _read_table(universe_dir / "category_nodes.parquet", ["category_id", "paper_count", "x", "y", "z"])

    px = _column_numpy(paper_nodes, "x")
    py = _column_numpy(paper_nodes, "y")
    pz = _column_numpy(paper_nodes, "z")
    primary_categories = _column_strings(paper_nodes, "primary_category")
    titles = _column_strings(paper_nodes, "title")
    category_ids = _column_strings(category_nodes, "category_id")
    category_counts = np.asarray(category_nodes.column("paper_count").to_pylist(), dtype=np.int64)
    cx = _column_numpy(category_nodes, "x")
    cy = _column_numpy(category_nodes, "y")
    cz = _column_numpy(category_nodes, "z")

    top_categories = [category_ids[idx] for idx in np.argsort(-category_counts)[: min(6, len(category_ids))]]
    palette = ["#2563eb", "#7c3aed", "#db2777", "#ea580c", "#0891b2", "#65a30d"]
    color_map = {category: palette[idx % len(palette)] for idx, category in enumerate(top_categories)}
    fallback_color = "#94a3b8"

    paper_idx = _sample_indices(len(px), paper_sample, seed + 1)
    sampled_colors = [color_map.get(primary_categories[idx], fallback_color) for idx in paper_idx.tolist()]
    category_centers = {
        category_ids[idx]: np.asarray([cx[idx], cy[idx], cz[idx]], dtype=np.float32) for idx in range(len(category_ids))
    }
    labeled_indices: list[int] = []
    used_indices: set[int] = set()
    for category in top_categories:
        candidates = [idx for idx in paper_idx.tolist() if primary_categories[idx] == category and titles[idx].strip()]
        if not candidates:
            continue
        center = category_centers.get(category)
        if center is None:
            chosen_idx = candidates[0]
        else:
            chosen_idx = min(
                candidates,
                key=lambda idx: float(
                    (px[idx] - center[0]) ** 2 + (py[idx] - center[1]) ** 2 + (pz[idx] - center[2]) ** 2
                ),
            )
        if chosen_idx in used_indices:
            continue
        labeled_indices.append(chosen_idx)
        used_indices.add(chosen_idx)

    fig = plt.figure(figsize=(15, 12), dpi=220, facecolor="#f8fafc")
    ax = fig.add_subplot(111, projection="3d")
    _style_axes(ax, title="Paper Universe 3D Detailed View")

    ax.scatter(
        px[paper_idx],
        py[paper_idx],
        pz[paper_idx],
        s=1.25,
        alpha=0.28,
        c=sampled_colors,
        linewidths=0,
        depthshade=False,
    )
    if labeled_indices:
        labeled_idx = np.asarray(labeled_indices, dtype=np.int64)
        ax.scatter(
            px[labeled_idx],
            py[labeled_idx],
            pz[labeled_idx],
            s=24,
            alpha=1.0,
            c=[color_map.get(primary_categories[idx], fallback_color) for idx in labeled_indices],
            edgecolors="white",
            linewidths=0.45,
            depthshade=False,
        )
        for idx in labeled_indices:
            ax.text(px[idx], py[idx], pz[idx], _truncate(titles[idx]), fontsize=7, color="#0f172a")

    category_count_map = {category_ids[idx]: int(category_counts[idx]) for idx in range(len(category_ids))}
    legend_lines = "\n".join(f"{cat}: {category_count_map.get(cat, 0):,} papers" for cat in top_categories)
    ax.text2D(
        0.02,
        0.98,
        f"paper nodes shown: {len(paper_idx):,}\nfocus categories:\n{legend_lines}\nlabeled papers: {len(labeled_indices):,}",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        color="#334155",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.4"},
    )

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "paper_points": int(len(paper_idx)),
        "highlighted_categories": int(len(top_categories)),
        "labeled_papers": int(len(labeled_indices)),
    }


def render_paper_universe_assets(
    *,
    universe_dir: str = str(DEFAULT_UNIVERSE_DIR),
    overview_sample: int = DEFAULT_OVERVIEW_SAMPLE,
    detailed_sample: int = DEFAULT_DETAILED_SAMPLE,
    seed: int = DEFAULT_SEED,
) -> Dict[str, object]:
    universe_root = Path(universe_dir).resolve()
    if not (universe_root / "paper_nodes.parquet").is_file():
        raise RuntimeError(f"Missing paper_nodes.parquet under {universe_root}.")
    if not (universe_root / "category_nodes.parquet").is_file():
        raise RuntimeError(f"Missing category_nodes.parquet under {universe_root}.")
    if not (universe_root / "year_nodes.parquet").is_file():
        raise RuntimeError(f"Missing year_nodes.parquet under {universe_root}.")

    overview_path = universe_root / "universe_3d.png"
    detailed_path = universe_root / "universe_3d_detailed.png"

    overview_stats = _render_overview(
        universe_root,
        paper_sample=int(overview_sample),
        seed=int(seed),
        output_path=overview_path,
    )
    detailed_stats = _render_detailed(
        universe_root,
        paper_sample=int(detailed_sample),
        seed=int(seed),
        output_path=detailed_path,
    )

    manifest = {
        "universe_dir": str(universe_root),
        "overview_image": str(overview_path),
        "detailed_image": str(detailed_path),
        "overview": overview_stats,
        "detailed": detailed_stats,
    }
    (universe_root / "render_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Render 3D PNG assets for the standalone paper universe.")
    parser.add_argument("--universe-dir", default=str(DEFAULT_UNIVERSE_DIR))
    parser.add_argument("--overview-sample", type=int, default=DEFAULT_OVERVIEW_SAMPLE)
    parser.add_argument("--detailed-sample", type=int, default=DEFAULT_DETAILED_SAMPLE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    result = render_paper_universe_assets(
        universe_dir=args.universe_dir,
        overview_sample=int(args.overview_sample),
        detailed_sample=int(args.detailed_sample),
        seed=int(args.seed),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
