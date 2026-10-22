"""Representación propia del grafo vial: listas de adyacencia dirigidas con tiempo (s) por arista."""
from dataclasses import dataclass, field


@dataclass
class Road:
    w: dict            # (u, v) -> tiempo de recorrido en segundos (arista dirigida)
    name: dict         # (u, v) -> nombre de la calle
    xy: dict           # nodo -> (lon, lat)
    adj: dict = field(default_factory=dict)  # u -> [(v, w)] ordenada por v

    def __post_init__(self):
        if not self.adj:
            for (u, v), t in sorted(self.w.items()):
                self.adj.setdefault(u, []).append((v, t))

    def subgraph(self, edges):
        """Sub-red dirigida inducida por un conjunto de aristas de self."""
        return Road({e: self.w[e] for e in edges}, {e: self.name[e] for e in edges}, self.xy)


def _as_str(x, default):
    if isinstance(x, list):
        x = x[0] if x else None
    return x if isinstance(x, str) and x else default


def from_osmnx(G):
    """Convierte el MultiDiGraph de OSMnx a Road: componente fuertemente conexa mayor,
    una sola arista por par (u, v) (la más rápida) y sin lazos."""
    import osmnx as ox
    G = ox.truncate.largest_component(G, strongly=True)
    w, name = {}, {}
    for u, v, d in G.edges(data=True):
        if u == v:
            continue
        t = float(d["travel_time"])
        if (u, v) not in w or t < w[(u, v)]:
            w[(u, v)] = t
            name[(u, v)] = _as_str(d.get("name"), "sin nombre (" + _as_str(d.get("highway"), "vía") + ")")
    xy = {n: (float(d["x"]), float(d["y"])) for n, d in G.nodes(data=True)}
    return Road(w, name, xy)
