"""Algoritmos propios: Dijkstra multi-fuente, Árbol de Steiner (Mehlhorn), Held–Karp y
ruta ordenada con reparación dirigida. Todos deterministas: los empates se rompen por id de nodo."""
import heapq
from math import inf


# ---------------------------------------------------------------- Dijkstra
def dijkstra(adj, sources, target=None):
    """Dijkstra multi-fuente con heap binario. Devuelve (dist, pred, root):
    root[v] = fuente más cercana a v (celda de Voronoi en el grafo)."""
    dist, pred, root, best = {}, {}, {}, {}
    heap = [(0.0, s, s) for s in sorted(set(sources))]
    heapq.heapify(heap)
    for s in sources:
        best[s] = 0.0
    while heap:
        d, u, r = heapq.heappop(heap)
        if u in dist:
            continue
        dist[u], root[u] = d, r
        if u == target:
            break
        for v, w in adj.get(u, ()):
            nd = d + w
            if v not in dist and nd < best.get(v, inf):
                best[v], pred[v] = nd, u
                heapq.heappush(heap, (nd, v, r))
    return dist, pred, root


def walk_back(pred, v):
    """Camino desde la fuente hasta v siguiendo predecesores."""
    path = [v]
    while v in pred:
        v = pred[v]
        path.append(v)
    return path[::-1]


# ---------------------------------------------------------------- Kruskal
class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        self.p[max(a, b)] = min(a, b)
        return True


def kruskal(weighted_edges):
    """weighted_edges: [(w, a, b)]. Devuelve las aristas (a, b) del bosque generador mínimo."""
    dsu = DSU()
    return [(a, b) for w, a, b in sorted(weighted_edges) if dsu.union(a, b)]


# ---------------------------------------------------------------- Steiner
def undirected(road, lam=0.0, tok=None):
    """Simetriza el grafo con peso w_λ(e) = t(e) + λ·tok(e), t = min de ambos sentidos."""
    wu = {}
    for (u, v), t in road.w.items():
        k = (min(u, v), max(u, v))
        wu[k] = min(wu.get(k, inf), t)
    if lam and tok:
        wu = {k: t + lam * tok[k] for k, t in wu.items()}
    adj = {}
    for (a, b), t in sorted(wu.items()):
        adj.setdefault(a, []).append((b, t))
        adj.setdefault(b, []).append((a, t))
    return adj, wu


def steiner_mehlhorn(adj, wu, terminals):
    """Árbol de Steiner 2(1-1/ℓ)-aproximado (Mehlhorn 1988) sobre el grafo no dirigido adj.
    Devuelve el conjunto de aristas no dirigidas (a, b), a < b."""
    terminals = sorted(set(terminals))
    if len(terminals) < 2:
        return set()
    # 1. Voronoi: un Dijkstra multi-fuente desde todos los terminales.
    dist, pred, root = dijkstra(adj, terminals)
    # 2. Grafo G1': por cada arista frontera entre celdas, conectar los terminales dueños.
    best = {}
    for (u, v), w in wu.items():
        if u in root and v in root and root[u] != root[v]:
            a, b = sorted((root[u], root[v]))
            cand = (dist[u] + w + dist[v], u, v)
            if (a, b) not in best or cand < best[(a, b)]:
                best[(a, b)] = cand
    mst = kruskal([(c[0], a, b) for (a, b), c in best.items()])
    if len(mst) != len(terminals) - 1:
        raise ValueError("terminales en componentes distintas")
    # 3. Expandir cada arista de G1' a su camino en G.
    edges = set()
    for a, b in mst:
        _, u, v = best[(a, b)]
        path = walk_back(pred, u) + walk_back(pred, v)[::-1]  # root[u] .. u, v .. root[v]
        for x, y in zip(path, path[1:]):
            edges.add((min(x, y), max(x, y)))
    # 4. MST del subgrafo expandido y 5. poda de hojas no terminales.
    tree = set(kruskal([(wu[e], *e) for e in edges]))
    return prune_leaves(tree, set(terminals))


def prune_leaves(tree, terminals):
    """Quita repetidamente hojas no terminales. O(|tree|)."""
    nb = {}
    for a, b in tree:
        nb.setdefault(a, set()).add(b)
        nb.setdefault(b, set()).add(a)
    leaves = sorted(x for x, s in nb.items() if len(s) == 1 and x not in terminals)
    tree = set(tree)
    while leaves:
        x = leaves.pop()
        if len(nb[x]) != 1:
            continue
        y = nb[x].pop()
        nb[y].discard(x)
        tree.discard((min(x, y), max(x, y)))
        if len(nb[y]) == 1 and y not in terminals:
            leaves.append(y)
    return tree


def tree_weight(tree, wu):
    return sum(wu[e] for e in tree)


# ---------------------------------------------------------------- Held–Karp
def held_karp(D, s, t, mids):
    """Orden óptimo s -> permutación(mids) -> t sobre la matriz de distancias D (dict).
    O(2^k k^2) con k = |mids|. Devuelve (costo, orden completo)."""
    k = len(mids)
    if k == 0:
        return D[s, t], [s, t]
    dp = {(1 << j, j): (D[s, mids[j]], None) for j in range(k)}
    for mask in range(1, 1 << k):
        for j in range(k):
            if (mask, j) not in dp:
                continue
            c = dp[mask, j][0]
            for n in range(k):
                if mask & (1 << n):
                    continue
                key, nc = (mask | (1 << n), n), c + D[mids[j], mids[n]]
                if key not in dp or nc < dp[key][0]:
                    dp[key] = (nc, j)
    full = (1 << k) - 1
    cost, last = min((dp[full, j][0] + D[mids[j], t], j) for j in range(k))
    order, mask = [], full
    while last is not None:
        order.append(mids[last])
        mask, last = mask ^ (1 << last), dp[mask, last][1]
    return cost, [s] + order[::-1] + [t]


# ---------------------------------------------------------------- Ruta ordenada
def ordered_route(road, sub_edges, s, t, pois, repair=True):
    """Ruta s -> POI (orden óptimo) -> t usando solo aristas dirigidas de sub_edges.
    Si falta un camino dirigido entre dos terminales (calles de un sentido), la reparación
    añade el camino mínimo en el grafo completo. Devuelve (ruta, costo, sub_edges, reparaciones)
    o None si no hay ruta y repair=False."""
    sub = set(sub_edges)
    terms = [s] + list(pois) + [t]
    D, P, repairs = {}, {}, 0
    for a in terms:
        sub_adj = road.subgraph(sub).adj
        dist, pred, _ = dijkstra(sub_adj, [a])
        for b in terms:
            if b == a:
                continue
            if b not in dist:
                if not repair:
                    return None
                fd, fp, _ = dijkstra(road.adj, [a], target=b)
                path = walk_back(fp, b)
                sub.update(zip(path, path[1:]))
                repairs += 1
                D[a, b], P[a, b] = fd[b], path
            else:
                D[a, b], P[a, b] = dist[b], walk_back(pred, b)
    cost, order = held_karp(D, s, t, list(pois))
    route = [s]
    for a, b in zip(order, order[1:]):
        route += P[a, b][1:]
    return route, cost, sub, repairs


def route_cost(road, route):
    return sum(road.w[e] for e in zip(route, route[1:]))


def is_valid_route(road, route, s, t, pois):
    return (route[0] == s and route[-1] == t and set(pois) <= set(route)
            and all(e in road.w for e in zip(route, route[1:])))


def tree_to_directed(road, tree):
    """Aristas dirigidas de G que realizan las aristas no dirigidas del árbol."""
    return {e for a, b in tree for e in ((a, b), (b, a)) if e in road.w}
