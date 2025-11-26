from __future__ import annotations

"""
Build multi-resolution (LOD) samples of the universe node embeddings for
WebGL-friendly visualization.

Inputs (from `scripts.universe_build`):
  - {export_root}/_universe/nodes.jsonl
  - {export_root}/_universe/node_coords.npy   (3D coords for each node)
  - Optional: {export_root}/_universe/edges.jsonl (for sampled edges)

Outputs (under {export_root}/_universe/lod/):
  - lod_<N>.jsonl : sampled nodes with fields {x,y,z,repo,kind,name,node_id}
  - edges_<N>.jsonl (optional) : edges where both endpoints are in the level
  - manifest.json : metadata about generated levels
  - viewer.html : lightweight deck.gl viewer that can switch LODs

Usage:
  python -m scripts.universe_lod \
    --export-root /data/repository_library/exports \
    --levels 10000,50000,200000 \
    --edges-level 50000 \
    --max-edges 200000

Serve the viewer (from _universe) with:
  cd /data/repository_library/exports/_universe
  python -m http.server 8000
  # open http://localhost:8000/lod/viewer.html
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

import numpy as np

UNIVERSE_DIRNAME = "_universe"
LOD_DIRNAME = "lod"


def _load_nodes(nodes_path: Path) -> List[Dict[str, str]]:
    nodes: List[Dict[str, str]] = []
    with nodes_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            n = json.loads(line)
            nodes.append(
                {
                    "repo": str(n.get("repo_id", "")),
                    "kind": str(n.get("kind", "")),
                    "name": str(n.get("name", "")),
                    "node_id": str(n.get("node_id", "")),
                }
            )
    return nodes


def _stratified_sample_indices(
    nodes: List[Dict[str, str]],
    sample_size: int,
) -> Set[int]:
    total = len(nodes)
    if sample_size >= total:
        return set(range(total))

    # Group indices by repo.
    per_repo: Dict[str, List[int]] = defaultdict(list)
    for idx, n in enumerate(nodes):
        per_repo[n["repo"]].append(idx)

    # Initial proportional allocation.
    alloc: Dict[str, int] = {}
    for repo, idxs in per_repo.items():
        share = max(1, int(len(idxs) / total * sample_size))
        alloc[repo] = min(share, len(idxs))

    current = sum(alloc.values())
    # Adjust down if we overshoot.
    while current > sample_size:
        # Remove 1 from the repo with the largest allocation that still has room to shrink.
        repo = max(alloc, key=lambda r: alloc[r])
        if alloc[repo] > 1:
            alloc[repo] -= 1
            current -= 1
        else:
            break
    # Adjust up if we undershoot.
    repos = list(per_repo.keys())
    while current < sample_size and repos:
        repo = random.choice(repos)
        if alloc[repo] < len(per_repo[repo]):
            alloc[repo] += 1
            current += 1
        else:
            repos.remove(repo)

    sampled: Set[int] = set()
    for repo, count in alloc.items():
        idxs = per_repo[repo]
        if count >= len(idxs):
            sampled.update(idxs)
        else:
            sampled.update(random.sample(idxs, count))
    return sampled


def _write_json_array(path: Path, rows: List[Dict[str, object]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def _build_edges_subset(
    edges_path: Path,
    keep_nodes: Set[str],
    max_edges: int,
    out_path: Path,
) -> int:
    kept = 0
    with edges_path.open("r", encoding="utf-8") as fh, out_path.open("w", encoding="utf-8") as out:
        for line in fh:
            if kept >= max_edges:
                break
            e = json.loads(line)
            if e.get("src") in keep_nodes and e.get("dst") in keep_nodes:
                out.write(json.dumps(e, ensure_ascii=False) + "\n")
                kept += 1
    return kept


def _write_viewer(lod_dir: Path, levels: List[int]) -> None:
    options = "".join([f'<option value="lod_{n}.json">lod_{n}</option>' for n in levels])
    html = r"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Repo Universe LOD Viewer</title>
  <style>
    body, html, #app { margin:0; padding:0; width:100%; height:100%; background: #0b0b0b; color: #eee; }
    #controls { position: absolute; top: 8px; left: 8px; z-index: 10; background: rgba(255,255,255,0.9); color:#000; padding: 6px 10px; border-radius: 6px; font-family: sans-serif; }
  </style>
  <script src="https://unpkg.com/deck.gl@latest/dist.min.js"></script>
  <script src="https://unpkg.com/@loaders.gl/core@latest/dist/dist.min.js"></script>
  <script src="https://unpkg.com/@loaders.gl/json@latest/dist/dist.min.js"></script>
</head>
<body>
  <div id="controls">
    <label for="level">LOD:</label>
    <select id="level">
      __LOD_OPTIONS__
    </select>
    <span id="status">loading…</span>
  </div>
  <div id="app"></div>
  <script>
    const {Deck, ScatterplotLayer, OrbitView, OrbitController, COORDINATE_SYSTEM} = deck;
    const levelSelect = document.getElementById('level');
    const status = document.getElementById('status');
    // Base path for the LOD files; keep relative to the viewer location.
    const lodRoot = './';
    let deckgl;
    let currentViewState = null;

    function hashColor(str) {
      let h = 0;
      for (let i = 0; i < str.length; i++) h = (h * 31 + str.charCodeAt(i)) | 0;
      const r = (h & 0xff0000) >> 16;
      const g = (h & 0x00ff00) >> 8;
      const b = (h & 0x0000ff);
      return [ (r % 200) + 30, (g % 200) + 30, (b % 200) + 30 ];
    }

    async function loadLevel(fname) {
      status.textContent = 'loading ' + fname + ' …';
      const url = lodRoot + fname;
      const resp = await fetch(url);
      if (!resp.ok) {
        throw new Error('fetch failed: ' + resp.status + ' ' + resp.statusText + ' for ' + url);
      }
      let rows = [];
      try {
        rows = await resp.json();
      } catch (e) {
        console.error('failed to parse JSON', e);
        throw e;
      }
      status.textContent = rows.length + ' nodes';
      return rows;
    }

    function computeViewState(rows) {
      if (!rows.length) return null;
      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
      for (const r of rows) {
        minX = Math.min(minX, r.x); maxX = Math.max(maxX, r.x);
        minY = Math.min(minY, r.y); maxY = Math.max(maxY, r.y);
        minZ = Math.min(minZ, r.z); maxZ = Math.max(maxZ, r.z);
      }
      const cx = (minX + maxX) / 2;
      const cy = (minY + maxY) / 2;
      const cz = (minZ + maxZ) / 2;
      const extent = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
      // Map extent to zoom; larger extent -> smaller zoom. Keep within visible bounds.
      const span = Math.max(extent, 1e-3);
      const zoom = Math.min(12, Math.max(0, Math.log2(256 / span)));
      const view = {id: 'orbit', target: [cx, cy, cz], zoom, rotationX: 45, rotationOrbit: 30, minZoom: -5, maxZoom: 20};
      console.log('bbox', {minX, maxX, minY, maxY, minZ, maxZ, extent, zoom});
      return view;
    }

    function render(rows) {
      const layer = new ScatterplotLayer({
        id: 'nodes',
        data: rows,
        getPosition: d => [d.x, d.y, d.z],
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        getRadius: 2,
        radiusUnits: 'pixels',
        radiusMinPixels: 4,
        radiusMaxPixels: 16,
        getFillColor: d => hashColor(d.repo || ''),
        opacity: 0.6,
        pickable: true,
        onHover: info => {
          const el = document.getElementById('status');
          if (info.object) {
            el.textContent = `${info.object.repo} | ${info.object.kind} | ${info.object.name}`;
          } else {
            el.textContent = rows.length + ' nodes';
          }
        },
      });
      if (!deckgl) {
        deckgl = new Deck({
          container: 'app',
          views: new OrbitView({id: 'orbit'}),
          controller: {type: OrbitController, dragRotate: true, inertia: true},
          initialViewState: currentViewState || {id:'orbit', target: [0,0,0], zoom: 4, rotationX: 45, rotationOrbit: 30},
          layers: [layer],
          parameters: {clearColor: [0.05, 0.05, 0.05, 1]},
          onViewStateChange: ({viewState}) => {
            currentViewState = viewState;
            deckgl.setProps({viewState});
          },
          onError: err => {
            console.error('deck error', err);
            status.textContent = 'deck error';
          },
        });
      } else {
        deckgl.setProps({layers: [layer], viewState: currentViewState || undefined});
      }
    }

    async function refresh() {
      const fname = levelSelect.value;
      try {
        const rows = await loadLevel(fname);
        if (!rows || !rows.length) {
          status.textContent = 'no data in ' + fname;
          if (deckgl) deckgl.setProps({layers: []});
          return;
        }
        currentViewState = computeViewState(rows);
        console.log('loaded', rows.length, 'nodes; viewState', currentViewState);
        render(rows);
      } catch (err) {
        console.error('refresh failed', err);
        status.textContent = 'error: ' + err;
        if (deckgl) deckgl.setProps({layers: []});
      }
    }

    levelSelect.addEventListener('change', refresh);
    refresh();
  </script>
</body>
</html>
"""
    html = html.replace("__LOD_OPTIONS__", options)
    (lod_dir / "viewer.html").write_text(html, encoding="utf-8")


def build_lod(
    export_root: Path,
    levels: List[int],
    edges_level: int | None,
    max_edges: int,
    separate_repos_scale: float = 0.0,
) -> None:
    uni_root = export_root / UNIVERSE_DIRNAME
    nodes_path = uni_root / "nodes.jsonl"
    coords_path = uni_root / "node_coords.npy"
    edges_path = uni_root / "edges.jsonl"
    repo_coords_path = uni_root / "repo_coords.npy"
    manifest_path = uni_root / "manifest.json"
    lod_dir = uni_root / LOD_DIRNAME
    lod_dir.mkdir(parents=True, exist_ok=True)

    nodes = _load_nodes(nodes_path)
    coords = np.load(coords_path)
    assert coords.shape[0] == len(nodes), "coords and nodes count mismatch"

    repo_offset: Dict[str, np.ndarray] = {}
    if separate_repos_scale > 0.0 and repo_coords_path.exists() and manifest_path.exists():
        try:
            repo_coords = np.load(repo_coords_path)
            man = json.loads(manifest_path.read_text())
            repo_ids = man.get("repo_ids") or []
            for idx, rid in enumerate(repo_ids):
                if idx < repo_coords.shape[0]:
                    repo_offset[str(rid)] = np.asarray(repo_coords[idx]) * separate_repos_scale
        except Exception:
            repo_offset = {}

    manifest = {"levels": []}
    for level in levels:
        sample_idx = _stratified_sample_indices(nodes, level)
        rows = []
        for idx in sample_idx:
            c = coords[idx]
            n = nodes[idx]
            offset = repo_offset.get(n["repo"])
            if offset is not None:
                c = c + offset
            rows.append(
                {
                    "x": float(c[0]),
                    "y": float(c[1]),
                    "z": float(c[2]),
                    "repo": n["repo"],
                    "kind": n["kind"],
                    "name": n["name"],
                    "node_id": n["node_id"],
                }
            )
        out_path = lod_dir / f"lod_{level}.json"
        _write_json_array(out_path, rows)

        edges_written = 0
        if edges_level and level == edges_level and edges_path.exists():
            keep_nodes = {r["node_id"] for r in rows}
            edges_out = lod_dir / f"edges_{level}.jsonl"
            edges_written = _build_edges_subset(edges_path, keep_nodes, max_edges, edges_out)

        manifest["levels"].append(
            {
                "level": level,
                "node_count": len(rows),
                "path": str(out_path.relative_to(uni_root)),
                "edges_path": f"lod/edges_{level}.jsonl" if edges_written else None,
                "edges_kept": edges_written,
            }
        )

    manifest_path = lod_dir / "manifest.json"
    manifest["built_from"] = {
        "nodes": str(nodes_path),
        "coords": str(coords_path),
        "repo_coords": str(repo_coords_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_viewer(lod_dir, levels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LOD samples for universe visualization.")
    parser.add_argument("--export-root", type=str, default="/data/repository_library/exports")
    parser.add_argument("--levels", type=str, default="10000,50000,200000", help="Comma-separated node counts per LOD.")
    parser.add_argument("--edges-level", type=int, default=50000, help="Which LOD to emit sampled edges for (0 to skip).")
    parser.add_argument("--max-edges", type=int, default=200000, help="Max edges to keep for the edges-level.")
    parser.add_argument("--separate-repos-scale", type=float, default=0.0, help="If >0, offset nodes by repo centroid scaled by this factor to spread repos apart.")
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    edges_level = int(args.edges_level) if args.edges_level and int(args.edges_level) > 0 else None
    build_lod(
        export_root=Path(args.export_root),
        levels=levels,
        edges_level=edges_level,
        max_edges=int(args.max_edges),
        separate_repos_scale=float(args.separate_repos_scale),
    )


if __name__ == "__main__":
    main()
