"""Figuras del reporte a partir de results/*.csv."""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "paper" / "fig"
COLORS = {"B0": "#7f7f7f", "B1": "#1f77b4", "B2": "#ff7f0e", "M": "#2ca02c", "REF": "#9467bd"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(RES / f"{name}.png", dpi=150)  # vista previa


def read(name):
    with open(RES / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tokens_by_zone(rows):
    zones = list(dict.fromkeys(r["zone"] for r in rows))
    n = {r["zone"]: int(r["n"]) for r in rows}
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for meth in ("B0", "B2", "M", "B1", "REF"):
        ys = [statistics.median(float(r["tokens"]) for r in rows if r["zone"] == z and r["method"] == meth)
              for z in zones]
        ax.plot([n[z] for z in zones], ys, "o-", label=meth, color=COLORS[meth], ms=3)
    ax.set(xscale="log", yscale="log", xlabel="nodos del grafo |V|", ylabel="tokens (mediana)")
    ax.legend(fontsize=6, ncol=3)
    ax.grid(alpha=.3, which="both", lw=.4)
    fig.tight_layout()
    save(fig, "tokens_zona")


def time_by_k(rows):
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for z in dict.fromkeys(r["zone"] for r in rows):
        R = [r for r in rows if r["zone"] == z and r["method"] == "M"]
        ks = sorted({int(r["k"]) for r in R})
        ax.plot(ks, [statistics.median(float(r["ms"]) for r in R if int(r["k"]) == k) for k in ks],
                "o-", ms=3, label=f"{z} (|V|={R[0]['n']})")
    ax.set(xlabel="POI por consulta (k)", ylabel="tiempo de M (ms, mediana)", yscale="log")
    ax.legend(fontsize=6)
    ax.grid(alpha=.3, which="both", lw=.4)
    fig.tight_layout()
    save(fig, "tiempo_k")


def pareto(rows):
    by = defaultdict(list)
    for r in rows:
        by[float(r["lam"])].append(r)
    lams = sorted(by)
    tok = [statistics.median(float(r["tokens"]) for r in by[l]) for l in lams]
    rho = [statistics.mean(float(r["rho"]) for r in by[l]) for l in lams]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(tok, rho, "o-", color=COLORS["M"], ms=3)
    for l, x, y in zip(lams, tok, rho):
        ax.annotate(f"λ={l:g}", (x, y), fontsize=5, xytext=(2, 2), textcoords="offset points")
    ax.set(xlabel="tokens del prompt (mediana)", ylabel="ρ medio (costo/óptimo)")
    ax.grid(alpha=.3, lw=.4)
    fig.tight_layout()
    save(fig, "pareto")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    rows = read("prelim.csv")
    tokens_by_zone(rows)
    time_by_k(rows)
    pareto(read("pareto.csv"))
    print("figuras en", FIG)
