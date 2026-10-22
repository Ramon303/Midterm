# steiner-llm

Equipo 7 · TC2038. Poda de grafos viales con Árbol de Steiner consciente de tokens para que un LLM
responda consultas de movilidad (origen, destino y varios POI) sin recibir el mapa completo.

## Estructura

- `src/road.py` — grafo dirigido propio (adyacencias + tiempo por arista), conversión desde OSMnx.
- `src/algorithms.py` — Dijkstra multi-fuente, Kruskal/DSU, Steiner de Mehlhorn, poda de hojas, Held–Karp, ruta ordenada con reparación dirigida.
- `src/serialize.py` — serialización (lista / JSON) con contracción de cadenas y conteo de tokens (`o200k_base`).
- `src/methods.py` — B0 grafo completo, B1 Dijkstra concatenado, B2 corredor tipo RAG, M propuesto, REF ruta óptima.
- `src/data.py` — descarga y cachea los grafos de OSM (`data/*.graphml`). OSMnx y NetworkX solo se usan para datos y como oráculo en pruebas.
- `tests/` — validación contra NetworkX y fuerza bruta.
- `experiments/run_prelim.py` — experimento preliminar (150 consultas fijas, semilla 2038). `experiments/plots.py` — figuras.
- `results/` — CSV de resultados. `paper/` — fuente LaTeX del Midterm 1 (IEEEtran).

## Reproducir

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m pytest
PYTHONIOENCODING=utf-8 .venv/Scripts/python experiments/run_prelim.py
.venv/Scripts/python experiments/plots.py
```

Los grafos de `data/` están versionados para que los datos no cambien entre corridas (OSM se edita a diario).
