"""Experimento preliminar (M1/M2): consultas fijas con semilla, cuatro métodos, sin LLM.
Mide tokens, tiempo, memoria, validez, ρ y determinismo. Salida: results/*.csv y figuras.
Uso: python experiments/run_prelim.py [--quick]""" 
import csv
import hashlib
import random
import statistics
import sys
import time
import tracemalloc
from math import inf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from algorithms import is_valid_route, route_cost, undirected, tree_weight  # noqa: E402
from data import load_graph  # noqa: E402
from methods import b0_full, b1_concat, b2_corridor, m_steiner, optimal_cost, ref_path  # noqa: E402
from road import from_osmnx  # noqa: E402
from serialize import count_tokens, edge_token_cost, serialize  # noqa: E402

SEED = 2038
ZONES = ("tec", "corredor", "centro")
KS = (1, 2, 3, 4, 6)          # número de POI
PER_K = 10                    # consultas por (zona, k)
REPS = 5                      # repeticiones para tiempo y determinismo
OUT = ROOT / "results"


def queries(road, zone):
    rng = random.Random(f"{SEED}-{zone}")
    nodes = sorted(road.xy)
    return [(k, rng.sample(nodes, k + 2)) for k in KS for _ in range(PER_K)]


def timed(fn, *args, **kw):
    times, outs = [], []
    for _ in range(REPS):
        t0 = time.perf_counter()
        outs.append(fn(*args, **kw))
        times.append(time.perf_counter() - t0)
    tracemalloc.start()
    fn(*args, **kw)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    digests = {hashlib.sha256(o["text"].encode()).hexdigest() for o in outs}
    outs[0]["sha"] = next(iter(digests))[:16]
    return outs[0], statistics.median(times), peak, len(digests) == 1


def main(quick=False):
    OUT.mkdir(exist_ok=True)
    rows = []
    for zone in ZONES[:2] if quick else ZONES:
        road = from_osmnx(load_graph(zone))
        tok = edge_token_cost(road)
        n, m = len(road.xy), len(road.w)
        qs = queries(road, zone)[: 10 if quick else None]
        b0_tokens = None
        for qi, (k, (s, t, *pois)) in enumerate(qs):
            opt = optimal_cost(road, s, t, pois)
            methods = {
                "B1": (b1_concat, {}),
                "B2": (b2_corridor, {}),
                "M": (m_steiner, {"tok": tok}),
                "REF": (ref_path, {}),
            }
            if b0_tokens is None:  # B0 no depende de la consulta salvo la cabecera
                out, sec, peak, det = timed(b0_full, road, s, t, pois)
                b0_tokens = count_tokens(out["text"])
                b0_json = count_tokens(serialize(road, road.w.keys(), s, t, pois, fmt="json"))
                b0_raw = count_tokens(serialize(road, road.w.keys(), s, t, pois, contract=False))
                rows.append(dict(zone=zone, n=n, m=m, q=qi, k=k, method="B0", tokens=b0_tokens,
                                 tokens_json=b0_json, tokens_nocontract=b0_raw, ms=sec * 1e3,
                                 peak_kb=peak / 1024, valid="", rho="", deterministic=det))
            for name, (fn, kw) in methods.items():
                out, sec, peak, det = timed(fn, road, s, t, pois, **kw)
                route = out["route"]
                valid = route is not None and is_valid_route(road, route, s, t, pois)
                rows.append(dict(
                    zone=zone, n=n, m=m, q=qi, k=k, method=name,
                    tokens=count_tokens(out["text"]),
                    tokens_json=count_tokens(serialize(road, out["edges"], s, t, pois, route, fmt="json")),
                    tokens_nocontract=count_tokens(serialize(road, out["edges"], s, t, pois, route, contract=False)),
                    ms=sec * 1e3, peak_kb=peak / 1024, valid=valid,
                    rho=route_cost(road, route) / opt if valid else "",
                    deterministic=det, repairs=out.get("repairs", ""), sha=out["sha"]))
            print(zone, qi, "ok", flush=True)
    _write(OUT / "prelim.csv", rows)
    pareto(quick)
    summary(rows)


def pareto(quick):
    """Frente λ: tokens vs. peso del árbol y ρ de la ruta, en el corredor."""
    road = from_osmnx(load_graph("corredor"))
    tok = edge_token_cost(road)
    rng = random.Random(f"{SEED}-pareto")
    rows = []
    for qi in range(5 if quick else 20):
        s, t, *pois = rng.sample(sorted(road.xy), 6)
        opt = optimal_cost(road, s, t, pois)
        _, base_wu = undirected(road)
        for lam in (0, 0.25, 0.5, 1, 2, 4, 8, 16, 32):
            out = m_steiner(road, s, t, pois, lambdas=(lam,), tok=tok)
            tree = out["tree"]
            rows.append(dict(q=qi, lam=lam, tokens=count_tokens(out["text"]),
                             tree_time=tree_weight(tree, base_wu), tree_edges=len(tree),
                             rho=route_cost(road, out["route"]) / opt))
    _write(OUT / "pareto.csv", rows)


def _write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summary(rows):
    print(f"\n{'zona':9}{'mét':4}{'tokens':>9}{'json':>9}{'sinContr':>9}{'ms':>9}{'KB':>9}{'válidas':>9}{'ρ med':>7}{'ρ max':>7}{'det':>5}")
    for zone in dict.fromkeys(r["zone"] for r in rows):
        for meth in ("B0", "B1", "B2", "M", "REF"):
            R = [r for r in rows if r["zone"] == zone and r["method"] == meth]
            if not R:
                continue
            med = lambda key: statistics.median(float(r[key]) for r in R)  # noqa: E731
            rhos = [r["rho"] for r in R if r["rho"] != ""]
            valid = sum(r["valid"] is True for r in R)
            print(f"{zone:9}{meth:4}{med('tokens'):9.0f}{med('tokens_json'):9.0f}{med('tokens_nocontract'):9.0f}"
                  f"{med('ms'):9.1f}{med('peak_kb'):9.0f}"
                  f"{(f'{valid}/{len(R)}' if meth != 'B0' else '-'):>9}"
                  f"{(statistics.median(rhos) if rhos else inf):7.3f}{(max(rhos) if rhos else inf):7.3f}"
                  f"{all(r['deterministic'] for r in R)!s:>5}")


if __name__ == "__main__":
    main("--quick" in sys.argv)
