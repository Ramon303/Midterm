"""Validación contra NetworkX (solo como oráculo) y fuerza bruta."""
import itertools
import random

import networkx as nx
import pytest

from algorithms import (dijkstra, undirected, steiner_mehlhorn, tree_weight, held_karp,
                        ordered_route, is_valid_route, tree_to_directed, route_cost)
from data import load_graph
from methods import m_steiner, b2_corridor, optimal_cost
from road import from_osmnx
from serialize import serialize, chains


@pytest.fixture(scope="module")
def road():
    return from_osmnx(load_graph("tec"))


def nx_digraph(road):
    D = nx.DiGraph()
    for (u, v), w in road.w.items():
        D.add_edge(u, v, weight=w)
    return D


def test_dijkstra_matches_networkx(road):
    D = nx_digraph(road)
    rng = random.Random(1)
    for s in rng.sample(sorted(road.xy), 10):
        dist, _, _ = dijkstra(road.adj, [s])
        ref = nx.single_source_dijkstra_path_length(D, s)
        assert dist.keys() == ref.keys()
        assert all(abs(dist[v] - ref[v]) < 1e-9 for v in ref)


def test_steiner_is_tree_spanning_terminals_and_matches_networkx(road):
    adj, wu = undirected(road)
    U = nx.Graph()
    for (a, b), w in wu.items():
        U.add_edge(a, b, weight=w)
    rng = random.Random(2)
    for k in (2, 3, 5, 8):
        terms = rng.sample(sorted(road.xy), k)
        tree = steiner_mehlhorn(adj, wu, terms)
        T = nx.Graph(list(tree))
        assert nx.is_tree(T) and set(terms) <= set(T)
        leaves = [n for n in T if T.degree(n) == 1]
        assert set(leaves) <= set(terms)
        ref = nx.algorithms.approximation.steiner_tree(U, terms, method="mehlhorn")
        # Misma garantía 2(1-1/ℓ); los empates pueden dar árboles distintos de peso parecido.
        assert tree_weight(tree, wu) <= 1.05 * ref.size(weight="weight")


def test_held_karp_matches_brute_force():
    rng = random.Random(3)
    nodes = list(range(7))
    D = {(a, b): rng.uniform(1, 10) for a in nodes for b in nodes if a != b}
    s, t, mids = 0, 6, nodes[1:6]
    cost, order = held_karp(D, s, t, mids)
    brute = min(sum(D[x] for x in zip((s, *p, t), (*p, t))) for p in itertools.permutations(mids))
    assert abs(cost - brute) < 1e-9
    assert sum(D[x] for x in zip(order, order[1:])) == pytest.approx(cost)


def test_m_route_valid_and_rho_at_least_one(road):
    rng = random.Random(4)
    for _ in range(10):
        s, t, *pois = rng.sample(sorted(road.xy), 5)
        out = m_steiner(road, s, t, pois)
        assert is_valid_route(road, out["route"], s, t, pois)
        assert route_cost(road, out["route"]) >= optimal_cost(road, s, t, pois) - 1e-9


def test_serialization_is_deterministic_and_contracts(road):
    s, t, *pois = sorted(road.xy)[:4]
    a = m_steiner(road, s, t, pois)["text"]
    b = m_steiner(road, s, t, pois)["text"]
    assert a == b
    full = list(road.w)
    assert len(chains(road, full, {s, t}, True)) < len(chains(road, full, {s, t}, False))


def test_corridor_may_fail_but_never_returns_invalid_route(road):
    rng = random.Random(5)
    for _ in range(10):
        s, t, *pois = rng.sample(sorted(road.xy), 4)
        out = b2_corridor(road, s, t, pois, delta=100)
        assert out["route"] is None or is_valid_route(road, out["route"], s, t, pois)
