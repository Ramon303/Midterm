"""Descarga y cachea el grafo vial de OSM. OSMnx solo se usa aquí, para obtener datos."""
from pathlib import Path

import osmnx as ox

CACHE = Path(__file__).resolve().parent.parent / "data"

# (oeste, sur, este, norte). Zonas crecientes para el experimento de escalabilidad.
ZONES = {
    "tec": (-100.300, 25.640, -100.280, 25.660),          # alrededor del Tec
    "corredor": (-100.320, 25.640, -100.280, 25.680),     # Tec -> Macroplaza
    "centro": (-100.360, 25.630, -100.260, 25.710),       # centro de Monterrey
}


def load_graph(zone: str):
    """MultiDiGraph dirigido con atributo travel_time (s) por arista."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{zone}.graphml"
    if path.exists():
        return ox.load_graphml(path)
    G = ox.graph_from_bbox(ZONES[zone], network_type="drive", simplify=True)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    ox.save_graphml(G, path)
    return G


if __name__ == "__main__":
    for z in ZONES:
        G = load_graph(z)
        print(z, G.number_of_nodes(), "nodos", G.number_of_edges(), "aristas")
