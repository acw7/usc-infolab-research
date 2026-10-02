# Predicts the most likely route between two lat/lon points from Veraset trip data.
#
# 1. build: every user's ping sequence becomes hex-to-hex hops; hops are counted and
#    turned into transition probabilities P(next=b | current=a), stored as -log(P).
# 2. predict: snap A and B to the nearest hex in the graph, then Dijkstra. Minimizing
#    the sum of -log(P) is the same as maximizing the product of P, so the result is
#    the most probable hop chain under a first-order Markov model.
# 3. serve: tiny HTTP wrapper around predict for other people to call.
from __future__ import annotations

import argparse
import heapq
import json
import math
import pickle
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import h3.api.basic_int as h3

from interesting_pairs_batch import (
    RES,
    TIME_THRESHOLD,
    build_user_sequences_from_prepared_frame,
    get_hex_lookup,
    list_parquet_files,
    prepare_sequence_frame,
    read_parquet_subset,
    LAT_COL,
    LON_COL,
    TIME_COL,
    USER_COL,
)

Graph = dict[int, dict[int, float]]  # cell -> {next_cell: -log P(next_cell | cell)}

MAX_SNAP_RINGS = 5  # ~1 km at res 9


def count_transitions(sequences, time_threshold: float, counts: dict[int, Counter]) -> None:
    lookup = get_hex_lookup()
    for seq in sequences.values():
        for i in range(len(seq.cells) - 1):
            # A big time gap means a new trip, so it is not a real hop.
            if seq.timestamps[i + 1] - seq.timestamps[i] > time_threshold:
                continue
            a = lookup.decode_cell(seq.cells[i])
            b = lookup.decode_cell(seq.cells[i + 1])
            counts[a][b] += 1


def counts_to_graph(counts: dict[int, Counter]) -> Graph:
    graph: Graph = {}
    for a, nexts in counts.items():
        total = sum(nexts.values())
        graph[a] = {b: -math.log(n / total) for b, n in nexts.items()}
    return graph


def build_graph(paths: list[Path], rows_per_file: int | None, time_threshold: float = TIME_THRESHOLD) -> Graph:
    counts: dict[int, Counter] = defaultdict(Counter)
    for path in paths:
        print(f"reading {path}")
        frame = read_parquet_subset(path, rows_per_file, [USER_COL, TIME_COL, LAT_COL, LON_COL])
        prepared, _ = prepare_sequence_frame(frame, RES)
        # Keep every user with at least 2 distinct hexes; no upper cap.
        sequences, _ = build_user_sequences_from_prepared_frame(prepared, 2, 10**9)
        count_transitions(sequences, time_threshold, counts)
    graph = counts_to_graph(counts)
    print(f"graph: {len(graph)} cells, {sum(map(len, graph.values()))} edges")
    return graph


def snap(graph_nodes: set[int], lat: float, lon: float) -> int | None:
    cell = h3.latlng_to_cell(lat, lon, RES)
    for k in range(MAX_SNAP_RINGS + 1):
        hits = [c for c in h3.grid_ring(cell, k) if c in graph_nodes]
        if hits:
            return min(hits, key=lambda c: h3.great_circle_distance(h3.cell_to_latlng(c), (lat, lon)))
    return None


def shortest_path(graph: Graph, start: int, goal: int) -> tuple[list[int], float] | None:
    dist = {start: 0.0}
    prev: dict[int, int] = {}
    heap = [(0.0, start)]
    while heap:
        d, cell = heapq.heappop(heap)
        if cell == goal:
            path = [cell]
            while path[-1] != start:
                path.append(prev[path[-1]])
            return path[::-1], d
        if d > dist[cell]:
            continue
        for nxt, w in graph.get(cell, {}).items():
            nd = d + w
            if nd < dist.get(nxt, math.inf):
                dist[nxt] = nd
                prev[nxt] = cell
                heapq.heappush(heap, (nd, nxt))
    return None


def predict_route(graph: Graph, lat_a: float, lon_a: float, lat_b: float, lon_b: float, nodes: set[int] | None = None):
    """Return (list of H3 cells, probability) or None if A, B, or a route between them is missing."""
    # ponytail: consecutive pings can skip hexes, so hops may jump between non-adjacent cells.
    # ponytail: single best path only; top-k needs Yen's algorithm or networkx.
    if nodes is None:
        nodes = all_nodes(graph)
    start, goal = snap(nodes, lat_a, lon_a), snap(nodes, lat_b, lon_b)
    if start is None or goal is None:
        return None
    found = shortest_path(graph, start, goal)
    if found is None:
        return None
    path, cost = found
    return path, math.exp(-cost)


def all_nodes(graph: Graph) -> set[int]:
    return set(graph) | {b for nexts in graph.values() for b in nexts}


def route_json(found) -> dict:
    path, prob = found
    return {
        "probability": prob,
        "path": [{"cell": h3.int_to_str(c), "lat": lat, "lon": lon} for c in path for lat, lon in [h3.cell_to_latlng(c)]],
    }


def parse_point(params: dict, lat_key: str, lon_key: str) -> tuple[float, float]:
    lat, lon = float(params[lat_key][0]), float(params[lon_key][0])
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("lat/lon out of range")
    return lat, lon


def serve(graph: Graph, host: str, port: int) -> None:
    nodes = all_nodes(graph)

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/health":
                return self.send_json(200, {"ok": True})
            if url.path != "/route":
                return self.send_json(404, {"error": "unknown path; use /route or /health"})
            params = parse_qs(url.query)
            try:
                lat_a, lon_a = parse_point(params, "lat_a", "lon_a")
                lat_b, lon_b = parse_point(params, "lat_b", "lon_b")
            except (KeyError, ValueError):
                return self.send_json(400, {"error": "need numeric lat_a, lon_a, lat_b, lon_b in valid range"})
            found = predict_route(graph, lat_a, lon_a, lat_b, lon_b, nodes)
            if found is None:
                return self.send_json(404, {"error": "no route found between these points"})
            self.send_json(200, route_json(found))

    print(f"serving on http://{host}:{port}  ({len(nodes)} cells loaded)")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def demo() -> None:
    # Two cells in a line, plus a detour. a->b->c seen 3x, a->d->c seen once.
    a, b, c, d = (h3.latlng_to_cell(34.02, -118.28 + 0.01 * i, RES) for i in range(4))
    counts = defaultdict(Counter)
    counts[a][b] += 3
    counts[a][d] += 1
    counts[b][c] += 3
    counts[d][c] += 1
    graph = counts_to_graph(counts)
    path, cost = shortest_path(graph, a, c)
    assert path == [a, b, c], path
    assert abs(math.exp(-cost) - 0.75) < 1e-9
    # A->C never observed as one trip, yet chained through b. Snapping from a nearby point works too.
    lat, lon = h3.cell_to_latlng(a)
    found = predict_route(graph, lat + 0.001, lon, *h3.cell_to_latlng(c))
    assert found and found[0][0] == a and found[0][-1] == c, found
    assert shortest_path(graph, c, a) is None
    print("demo ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build graph from a parquet file or directory")
    b.add_argument("data", type=Path)
    b.add_argument("out", type=Path)
    b.add_argument("--rows-per-file", type=int, default=5_000_000, help="0 = all rows")
    b.add_argument("--max-files", type=int, default=0, help="0 = all files")
    p = sub.add_parser("predict", help="predict one route locally")
    p.add_argument("graph", type=Path)
    p.add_argument("coords", type=float, nargs=4, metavar=("LAT_A", "LON_A", "LAT_B", "LON_B"))
    s = sub.add_parser("serve", help="serve /route over HTTP")
    s.add_argument("graph", type=Path)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    sub.add_parser("demo", help="run self-check")
    args = parser.parse_args()

    if args.cmd == "demo":
        return demo()
    if args.cmd == "build":
        files = [args.data] if args.data.is_file() else list_parquet_files(args.data, None, args.max_files)
        graph = build_graph(files, args.rows_per_file or None)
        args.out.write_bytes(pickle.dumps(graph))
        return print(f"wrote {args.out}")

    graph = pickle.loads(args.graph.read_bytes())
    if args.cmd == "predict":
        found = predict_route(graph, *args.coords)
        print(json.dumps(route_json(found), indent=2) if found else "no route found")
    else:
        serve(graph, args.host, args.port)


if __name__ == "__main__":
    main()
