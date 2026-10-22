"""Serialización del subgrafo para el prompt y conteo de tokens.
Contracción de cadenas: un nodo no terminal de grado 2 cuyas dos aristas son de la misma calle
y del mismo sentido no aporta información de navegación; se funde en una sola línea."""
import json
from functools import lru_cache

import tiktoken

_ENC = tiktoken.get_encoding("o200k_base")


def count_tokens(text):
    return len(_ENC.encode(text))


@lru_cache(maxsize=None)
def name_tokens(name):
    return count_tokens(name)


def edge_token_cost(road):
    """tok(e) para el peso w_λ: tokens de una línea sin contraer (cota superior por arista)."""
    tok = {}
    for (u, v), n in road.name.items():
        k = (min(u, v), max(u, v))
        tok[k] = max(tok.get(k, 0), name_tokens(n) + 6)  # ~6 tokens de ids, flecha y tiempo
    return tok


def _segments(road, edges):
    """Aristas dirigidas -> segmentos no dirigidos {(a,b): doble_sentido}."""
    seg = {}
    for u, v in edges:
        k = (min(u, v), max(u, v))
        seg[k] = seg.get(k, False) or ((v, u) in edges)
    return seg


def chains(road, edges, terminals, contract=True):
    """Devuelve [(nodos, calle, doble, segundos)] en orden determinista."""
    edges = set(edges)
    seg = _segments(road, edges)
    nb = {}
    for a, b in seg:
        nb.setdefault(a, []).append(b)
        nb.setdefault(b, []).append(a)

    def ename(a, b):
        return road.name[(a, b)] if (a, b) in edges else road.name[(b, a)]

    def interior(x):
        if not contract or x in terminals or len(nb[x]) != 2:
            return False
        p, q = nb[x]
        sp, sq = (min(p, x), max(p, x)), (min(q, x), max(q, x))
        if ename(p, x) != ename(x, q) or seg[sp] != seg[sq]:
            return False
        return seg[sp] or ((p, x) in edges) != ((q, x) in edges)  # un sentido: entra y sale

    out, seen = [], set()

    def walk(a, b):
        nodes = [a, b]
        seen.add((min(a, b), max(a, b)))
        while interior(nodes[-1]):
            x = nodes[-1]
            nxt = [y for y in nb[x] if y != nodes[-2]][0]
            if (min(x, nxt), max(x, nxt)) in seen:
                break
            seen.add((min(x, nxt), max(x, nxt)))
            nodes.append(nxt)
        doble = seg[(min(a, b), max(a, b))]
        if not doble and (a, b) not in edges:
            nodes.reverse()  # orientar en el sentido permitido
        secs = sum(road.w[e] if e in edges else road.w[e[::-1]] for e in zip(nodes, nodes[1:]))
        out.append((nodes, ename(nodes[0], nodes[1]), doble, secs))

    for x in sorted(nb):
        if not interior(x):
            for y in sorted(nb[x]):
                if (min(x, y), max(x, y)) not in seen:
                    walk(x, y)
    for a, b in sorted(seg):  # ciclos formados solo por nodos interiores
        if (a, b) not in seen:
            walk(a, b)
    return out


def serialize(road, edges, s, t, pois, route=None, fmt="lista", contract=True):
    """Texto que recibe el LLM. Los ids OSM (10 dígitos) se renombran a ids locales cortos."""
    terminals = [s, t] + list(pois)
    ch = chains(road, edges, set(terminals), contract)
    ids = {n: i for i, n in enumerate(terminals)}
    for nodes, *_ in ch:
        for n in (nodes[0], nodes[-1]):
            ids.setdefault(n, len(ids))
    head = {"origen": 0, "destino": 1, "poi": list(range(2, 2 + len(pois)))}
    r = [ids[n] for n in route if n in ids] if route else None
    if fmt == "json":
        calles = [{"calle": nm, "de": ids[n[0]], "a": ids[n[-1]], "doble": d, "seg": round(sec)}
                  for n, nm, d, sec in ch]
        return json.dumps({**head, "ruta": r, "calles": calles}, ensure_ascii=False, separators=(",", ":"))
    lines = [f"O={head['origen']} D={head['destino']} P={','.join(map(str, head['poi']))}"]
    if r:
        lines.append("ruta:" + ">".join(map(str, r)))
    lines += [f"{nm}|{ids[n[0]]}{'<>' if d else '>'}{ids[n[-1]]}|{round(sec)}" for n, nm, d, sec in ch]
    return "\n".join(lines)
