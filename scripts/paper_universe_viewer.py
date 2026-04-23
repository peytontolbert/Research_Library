from __future__ import annotations

"""
Build a local interactive HTML viewer for the paper universe.

Outputs under the paper universe directory:

- universe_3d_hover.html
- nodes_3d_sample.html
- interactive/manifest.json
- interactive/papers_<N>.json
- interactive/categories.json
- interactive/years.json
"""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

import pyarrow.parquet as pq  # type: ignore


DEFAULT_UNIVERSE_DIR = Path("/data/repository_library/exports/_paper_universe")
DEFAULT_LEVELS = (50000, 200000)
DEFAULT_BATCH_ROWS = 8192
INTERACTIVE_DIRNAME = "interactive"


def _string(value: Any) -> str:
    return str(value or "").strip()


def _sample_positions(total_rows: int, sample_size: int, seed: int) -> Set[int]:
    if total_rows <= 0:
        return set()
    cap = min(max(1, int(sample_size or 0)), total_rows)
    rng = random.Random(int(seed))
    positions = rng.sample(range(total_rows), cap) if cap < total_rows else list(range(total_rows))
    return set(positions)


def _iter_sampled_rows(
    parquet_path: Path,
    *,
    columns: Sequence[str],
    positions: Set[int],
    batch_rows: int,
) -> List[Dict[str, Any]]:
    parquet_file = pq.ParquetFile(parquet_path)
    selected: List[Dict[str, Any]] = []
    offset = 0
    if not positions:
        return selected
    sorted_positions = sorted(int(pos) for pos in positions)
    pointer = 0
    for batch in parquet_file.iter_batches(columns=list(columns), batch_size=max(1, int(batch_rows or 1))):
        if pointer >= len(sorted_positions):
            break
        end = offset + batch.num_rows
        if sorted_positions[pointer] >= end:
            offset = end
            continue
        rows = batch.to_pylist()
        while pointer < len(sorted_positions) and sorted_positions[pointer] < end:
            local_idx = int(sorted_positions[pointer] - offset)
            if 0 <= local_idx < len(rows):
                selected.append(rows[local_idx])
            pointer += 1
        offset = end
    return selected


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _paper_rows_for_level(universe_dir: Path, *, sample_size: int, seed: int, batch_rows: int) -> List[Dict[str, Any]]:
    paper_path = universe_dir / "paper_nodes.parquet"
    total_rows = int(pq.ParquetFile(paper_path).metadata.num_rows)
    positions = _sample_positions(total_rows, sample_size, seed)
    rows = _iter_sampled_rows(
        paper_path,
        columns=["paper_id", "canonical_paper_id", "title", "primary_category", "year", "x", "y", "z"],
        positions=positions,
        batch_rows=batch_rows,
    )
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "paper_id": _string(row.get("paper_id")),
                "canonical_paper_id": _string(row.get("canonical_paper_id")),
                "title": _string(row.get("title")),
                "primary_category": _string(row.get("primary_category")),
                "year": int(row.get("year") or 0),
                "x": float(row.get("x") or 0.0),
                "y": float(row.get("y") or 0.0),
                "z": float(row.get("z") or 0.0),
            }
        )
    return out


def _table_rows(path: Path, columns: Sequence[str]) -> List[Dict[str, Any]]:
    return pq.read_table(path, columns=list(columns)).to_pylist()


def _category_rows(universe_dir: Path) -> List[Dict[str, Any]]:
    rows = _table_rows(universe_dir / "category_nodes.parquet", ["category_id", "paper_count", "x", "y", "z"])
    return [
        {
            "category_id": _string(row.get("category_id")),
            "paper_count": int(row.get("paper_count") or 0),
            "x": float(row.get("x") or 0.0),
            "y": float(row.get("y") or 0.0),
            "z": float(row.get("z") or 0.0),
        }
        for row in rows
    ]


def _year_rows(universe_dir: Path) -> List[Dict[str, Any]]:
    rows = _table_rows(universe_dir / "year_nodes.parquet", ["year", "paper_count", "x", "y", "z"])
    return [
        {
            "year": int(row.get("year") or 0),
            "paper_count": int(row.get("paper_count") or 0),
            "x": float(row.get("x") or 0.0),
            "y": float(row.get("y") or 0.0),
            "z": float(row.get("z") or 0.0),
        }
        for row in rows
    ]


def _write_viewer_html(universe_dir: Path, asset_dir: str) -> str:
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Paper Universe 3D Viewer</title>
  <style>
    html, body {{ margin:0; padding:0; width:100%; height:100%; overflow:hidden; background:#07111f; color:#e2e8f0; }}
    #app {{ position:fixed; inset:0; }}
    #controls {{
      position: fixed;
      top: 10px;
      left: 10px;
      z-index: 10;
      background: rgba(255,255,255,0.92);
      color:#0f172a;
      padding: 10px 12px;
      border-radius: 10px;
      font-family: sans-serif;
      display:flex;
      flex-wrap:wrap;
      gap:8px;
      align-items:center;
      max-width: calc(100vw - 24px);
      box-shadow: 0 18px 45px rgba(2, 8, 23, 0.20);
    }}
    #status {{
      min-width: 280px;
      color:#334155;
    }}
    select, input {{
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      padding: 4px 6px;
      background: white;
    }}
    .hint {{
      color:#475569;
      font-size: 12px;
    }}
  </style>
  <script src="https://unpkg.com/deck.gl@latest/dist.min.js"></script>
</head>
<body>
  <div id="controls">
    <label for="viewMode">View:</label>
    <select id="viewMode">
      <option value="papers">Papers</option>
      <option value="categories">Categories</option>
      <option value="papers+categories" selected>Papers + Categories</option>
      <option value="all">All</option>
    </select>
    <label for="paperLevel">Paper LOD:</label>
    <select id="paperLevel"></select>
    <label for="colorMode">Color:</label>
    <select id="colorMode">
      <option value="category" selected>Primary Category</option>
      <option value="year">Year</option>
      <option value="uniform">Uniform</option>
    </select>
    <label for="categoryFilter">Filter Category:</label>
    <select id="categoryFilter">
      <option value="">All Categories</option>
    </select>
    <span class="hint">Click a category anchor to filter papers.</span>
    <span id="status">loading…</span>
  </div>
  <div id="app"></div>
  <script>
    const {{Deck, ScatterplotLayer, TextLayer, OrbitView, OrbitController}} = deck;
    const assetRoot = './{asset_dir}/';
    const viewModeEl = document.getElementById('viewMode');
    const paperLevelEl = document.getElementById('paperLevel');
    const colorModeEl = document.getElementById('colorMode');
    const categoryFilterEl = document.getElementById('categoryFilter');
    const statusEl = document.getElementById('status');
    let deckgl = null;
    let manifest = null;
    let cached = {{}};
    let currentViewState = null;

    function hashColor(str) {{
      let h = 0;
      for (let i = 0; i < str.length; i++) h = (h * 31 + str.charCodeAt(i)) | 0;
      const r = (Math.abs(h) % 160) + 60;
      const g = (Math.abs(h >> 8) % 160) + 60;
      const b = (Math.abs(h >> 16) % 160) + 60;
      return [r, g, b];
    }}

    function yearColor(year) {{
      const y = Number(year || 0);
      const base = ((y % 17) * 13) % 255;
      return [40 + (base % 180), 120 + ((base * 2) % 100), 255 - (base % 120)];
    }}

    function paperColor(row) {{
      if (colorModeEl.value === 'uniform') return [29, 78, 216];
      if (colorModeEl.value === 'year') return yearColor(row.year);
      return hashColor(row.primary_category || 'unknown');
    }}

    function categoryColor(row) {{
      return hashColor(row.category_id || 'category');
    }}

    async function loadJson(path) {{
      if (!cached[path]) {{
        cached[path] = fetch(path).then(r => r.json());
      }}
      return cached[path];
    }}

    async function ensureManifest() {{
      if (!manifest) {{
        manifest = await loadJson(assetRoot + 'manifest.json');
        for (const level of manifest.paper_levels) {{
          const opt = document.createElement('option');
          opt.value = level.path;
          opt.textContent = level.label;
          paperLevelEl.appendChild(opt);
        }}
        const cats = await loadJson(assetRoot + manifest.categories.path);
        for (const row of cats) {{
          const opt = document.createElement('option');
          opt.value = row.category_id;
          opt.textContent = `${{row.category_id}} (${{row.paper_count.toLocaleString()}})`;
          categoryFilterEl.appendChild(opt);
        }}
      }}
      return manifest;
    }}

    function ensureDeck() {{
      if (deckgl) return deckgl;
        deckgl = new Deck({{
          parent: document.getElementById('app'),
          views: [new OrbitView({{orbitAxis: 'Z'}})],
          controller: new OrbitController(),
          initialViewState: {{target: [0, 0, 0], rotationX: 25, rotationOrbit: 35, zoom: 0.2}},
          getTooltip: ({{object, layer}}) => {{
          if (!object) return null;
          if (layer && layer.id === 'papers') {{
            return {{
              html: `<b>${{object.title || object.canonical_paper_id}}</b><br/>${{object.primary_category || 'unknown'}}<br/>year: ${{object.year || 'n/a'}}`,
              style: {{backgroundColor: 'rgba(15, 23, 42, 0.92)', color: '#f8fafc'}}
            }};
          }}
          if (layer && layer.id === 'categories') {{
            return {{
              html: `<b>${{object.category_id}}</b><br/>papers: ${{(object.paper_count || 0).toLocaleString()}}`,
              style: {{backgroundColor: 'rgba(15, 23, 42, 0.92)', color: '#f8fafc'}}
            }};
          }}
          if (layer && layer.id === 'years') {{
            return {{
              html: `<b>${{object.year}}</b><br/>papers: ${{(object.paper_count || 0).toLocaleString()}}`,
              style: {{backgroundColor: 'rgba(15, 23, 42, 0.92)', color: '#f8fafc'}}
            }};
          }}
          return null;
        }}
      }});
      return deckgl;
    }}

    function maybeResetView(rows) {{
      if (currentViewState || !rows.length) return;
      const xs = rows.map(r => r.x);
      const ys = rows.map(r => r.y);
      const zs = rows.map(r => r.z);
      const center = [
        (Math.min(...xs) + Math.max(...xs)) / 2,
        (Math.min(...ys) + Math.max(...ys)) / 2,
        (Math.min(...zs) + Math.max(...zs)) / 2
      ];
      currentViewState = {{target: center, rotationX: 25, rotationOrbit: 35, zoom: 0.2}};
      ensureDeck().setProps({{initialViewState: currentViewState}});
    }}

    async function render() {{
      const man = await ensureManifest();
      const paperPath = assetRoot + paperLevelEl.value;
      const papersRaw = await loadJson(paperPath);
      const categories = await loadJson(assetRoot + man.categories.path);
      const years = await loadJson(assetRoot + man.years.path);
      const activeView = viewModeEl.value;
      const categoryFilter = categoryFilterEl.value;
      const papers = categoryFilter ? papersRaw.filter(r => r.primary_category === categoryFilter) : papersRaw;
      maybeResetView(papers.length ? papers : categories);

      const showPapers = activeView === 'papers' || activeView === 'papers+categories' || activeView === 'all';
      const showCategories = activeView === 'categories' || activeView === 'papers+categories' || activeView === 'all';
      const showYears = activeView === 'all';

      const layers = [];
      if (showPapers) {{
        layers.push(new ScatterplotLayer({{
          id: 'papers',
          data: papers,
          pickable: true,
          opacity: 0.28,
          radiusUnits: 'pixels',
          getPosition: d => [d.x, d.y, d.z],
          getRadius: _ => 1.1,
          getFillColor: d => [...paperColor(d), 190],
          radiusMinPixels: 1,
          radiusMaxPixels: 3
        }}));
      }}
      if (showCategories) {{
        layers.push(new ScatterplotLayer({{
          id: 'categories',
          data: categories,
          pickable: true,
          opacity: 0.95,
          radiusUnits: 'pixels',
          getPosition: d => [d.x, d.y, d.z],
          getRadius: d => Math.max(6, Math.min(18, 6 + Math.log10((d.paper_count || 1) + 1) * 3)),
          getFillColor: d => [...categoryColor(d), 235],
          onClick: info => {{
            if (info.object) {{
              categoryFilterEl.value = info.object.category_id || '';
              render();
            }}
          }}
        }}));
        layers.push(new TextLayer({{
          id: 'category-labels',
          data: categories.slice(0, 48),
          pickable: false,
          getPosition: d => [d.x, d.y, d.z],
          getText: d => d.category_id,
          getColor: _ => [15, 23, 42, 255],
          getSize: _ => 14,
          sizeUnits: 'pixels',
          getTextAnchor: _ => 'start',
          getAlignmentBaseline: _ => 'center'
        }}));
      }}
      if (showYears) {{
        layers.push(new ScatterplotLayer({{
          id: 'years',
          data: years,
          pickable: true,
          opacity: 0.95,
          radiusUnits: 'pixels',
          getPosition: d => [d.x, d.y, d.z],
          getRadius: _ => 8,
          getFillColor: d => [...yearColor(d.year), 240]
        }}));
        layers.push(new TextLayer({{
          id: 'year-labels',
          data: years,
          pickable: false,
          getPosition: d => [d.x, d.y, d.z],
          getText: d => String(d.year),
          getColor: _ => [6, 78, 59, 255],
          getSize: _ => 14,
          sizeUnits: 'pixels',
          getTextAnchor: _ => 'middle',
          getAlignmentBaseline: _ => 'center'
        }}));
      }}

      ensureDeck().setProps({{
        layers,
        onViewStateChange: ({{viewState}}) => {{
          currentViewState = viewState;
        }}
      }});
      statusEl.textContent = `papers shown: ${{papers.length.toLocaleString()}} | categories: ${{categories.length.toLocaleString()}} | years: ${{years.length.toLocaleString()}}`;
    }}

    viewModeEl.addEventListener('change', render);
    paperLevelEl.addEventListener('change', render);
    colorModeEl.addEventListener('change', render);
    categoryFilterEl.addEventListener('change', render);
    ensureManifest().then(() => render());
  </script>
</body>
</html>
"""
    hover_path = universe_dir / "universe_3d_hover.html"
    sample_path = universe_dir / "nodes_3d_sample.html"
    hover_path.write_text(html, encoding="utf-8")
    sample_path.write_text(html, encoding="utf-8")
    return str(hover_path)


def build_paper_universe_viewer(
    *,
    universe_dir: str = str(DEFAULT_UNIVERSE_DIR),
    levels: Sequence[int] = DEFAULT_LEVELS,
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> Dict[str, Any]:
    universe_root = Path(universe_dir).resolve()
    interactive_dir = universe_root / INTERACTIVE_DIRNAME
    interactive_dir.mkdir(parents=True, exist_ok=True)

    level_entries: List[Dict[str, Any]] = []
    for idx, level in enumerate(levels):
        filename = f"papers_{int(level)}.json"
        rows = _paper_rows_for_level(
            universe_root,
            sample_size=int(level),
            seed=42 + idx,
            batch_rows=int(batch_rows),
        )
        _write_json(interactive_dir / filename, rows)
        level_entries.append(
            {
                "label": f"{len(rows):,} papers",
                "rows": len(rows),
                "path": filename,
            }
        )

    categories = _category_rows(universe_root)
    years = _year_rows(universe_root)
    _write_json(interactive_dir / "categories.json", categories)
    _write_json(interactive_dir / "years.json", years)
    manifest = {
        "built_at": int(time.time()),
        "paper_levels": level_entries,
        "categories": {"rows": len(categories), "path": "categories.json"},
        "years": {"rows": len(years), "path": "years.json"},
    }
    _write_json(interactive_dir / "manifest.json", manifest)
    html_path = _write_viewer_html(universe_root, INTERACTIVE_DIRNAME)
    result = {
        "universe_dir": str(universe_root),
        "interactive_dir": str(interactive_dir),
        "html_path": html_path,
        "paper_levels": level_entries,
        "category_rows": len(categories),
        "year_rows": len(years),
    }
    (universe_root / "viewer_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an interactive HTML viewer for the paper universe.")
    parser.add_argument("--universe-dir", default=str(DEFAULT_UNIVERSE_DIR))
    parser.add_argument("--levels", default="50000,200000", help="Comma-separated paper sample sizes.")
    parser.add_argument("--batch-rows", type=int, default=DEFAULT_BATCH_ROWS)
    args = parser.parse_args()
    levels = [int(part) for part in str(args.levels).split(",") if str(part).strip()]
    result = build_paper_universe_viewer(
        universe_dir=args.universe_dir,
        levels=levels,
        batch_rows=int(args.batch_rows),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
