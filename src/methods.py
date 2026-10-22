"""Métodos comparados. Cada uno devuelve el texto que iría al prompt y, si aplica, la ruta.
B0 grafo completo · B1 caminos mínimos concatenados en el orden de la consulta ·
B2 recuperación local tipo RAG (corredor espacial) · M Steiner consciente de tokens + ruta ordenada."""
from math import cos, radians, inf

from algorithms import (dijkstra, walk_back, undirected, steiner_mehlhorn, ordered_route,
                        tree_to_directed, route_cost)
from serialize import serialize, count_tokens, edge_token_cost

LAMBDAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def b0_full(road, s, t, pois):
    return {"text": serialize(road, road.w.keys(), s, t, pois), "route": None, "edges": set(road.w)}


def b1_concat(road, s, t, pois):
    """Dijkstra tramo por tramo en el orden en que el usuario nombró los POI (sin optimizar el orden)."""
    seq, route = [s] + list(pois) + [t], [s]
    for a, b in zip(seq, seq[1:]):
        dist, pred, _ = dijkstra(road.adj, [a], target=b)
        route += walk_back(pred, b)[1:]
    edges = set(zip(route, route[1:]))
    return {"text": serialize(road, edges, s, t, pois, route), "route": route, "edges": edges}


def _meters(p, q, lat0):
    k = 111_320.0
    return ((p[0] - q[0]) * k * cos(radians(lat0)), (p[1] - q[1]) * k)


def _dist_to_segment(p, a, b, lat0):
    ax, ay = _meters(a, p, lat0)
    bx, by = _meters(b, p, lat0)
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    u = 0.0 if L == 0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / L))
    x, y = ax + u * dx, ay + u * dy
    return (x * x + y * y) ** 0.5


def b2_corridor(road, s, t, pois, delta=250.0):
    """Recupera las calles a menos de delta metros de la poligonal s -> POI -> t (orden de la consulta).
    No garantiza conectividad: si el subgrafo no conecta los terminales la ruta es inválida."""
    seq = [road.xy[n] for n in [s] + list(pois) + [t]]
    lat0 = seq[0][1]
    near = {n for n, p in road.xy.items()
            if min(_dist_to_segment(p, a, b, lat0) for a, b in zip(seq, seq[1:])) <= delta}
    edges = {e for e in road.w if e[0] in near and e[1] in near}
    res = ordered_route(road, edges, s, t, pois, repair=False)
    route = res[0] if res else None
    return {"text": serialize(road, edges, s, t, pois), "route": route, "edges": edges}


def m_steiner(road, s, t, pois, budget=inf, lambdas=LAMBDAS, tok=None):
    """Método propuesto. Barre λ en orden creciente y devuelve el primer resultado cuyo texto
    cabe en el presupuesto B (el de menor λ, es decir, el que más prioriza el costo de ruta)."""
    tok = tok or edge_token_cost(road)
    out = None
    for lam in lambdas:
        adj, wu = undirected(road, lam, tok)
        tree = steiner_mehlhorn(adj, wu, [s, t, *pois])
        route, _, sub, repairs = ordered_route(road, tree_to_directed(road, tree), s, t, pois)
        text = serialize(road, sub, s, t, pois, route)
        out = {"text": text, "route": route, "edges": sub, "tree": tree, "lam": lam, "repairs": repairs}
        if count_tokens(text) <= budget:
            out["fits"] = True
            return out
    out["fits"] = False
    return out


def optimal_route(road, s, t, pois):
    """Referencia exacta: Held–Karp sobre distancias dirigidas de Dijkstra en el grafo completo.
    Devuelve (costo, ruta)."""
    from algorithms import held_karp
    terms = [s] + list(pois) + [t]
    D, P = {}, {}
    for a in terms:
        dist, pred, _ = dijkstra(road.adj, [a])
        for b in terms:
            if b != a:
                D[a, b], P[a, b] = dist.get(b, inf), walk_back(pred, b)
    cost, order = held_karp(D, s, t, list(pois))
    route = [s]
    for a, b in zip(order, order[1:]):
        route += P[a, b][1:]
    return cost, route


def optimal_cost(road, s, t, pois):
    return optimal_route(road, s, t, pois)[0]


def ref_path(road, s, t, pois):
    """Respuesta de referencia: solo la ruta óptima. Cota inferior de tokens; sin red alrededor."""
    route = optimal_route(road, s, t, pois)[1]
    edges = set(zip(route, route[1:]))
    return {"text": serialize(road, edges, s, t, pois, route), "route": route, "edges": edges}


__all__ = ["b0_full", "b1_concat", "b2_corridor", "m_steiner", "optimal_cost", "optimal_route", "ref_path", "route_cost"]
