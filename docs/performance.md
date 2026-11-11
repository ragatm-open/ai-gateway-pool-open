# Performance

## Budgets

| Mesure | Budget (fixé avant mesure) |
|---|---|
| Surcoût du Pool sur une requête non inférentielle (`/health/runtime`) | < 5 ms p50 |
| Surcoût sur le premier événement d'un flux | < 50 ms |
| Transformation du corps ou du flux | aucune (octets identiques) |

## Résultats — Gateway Open réelle, CPU, un nœud (2026-11-09)

Environnement : `deploy/compose/qualification/` sur un poste de développement (i5-1145G7, 8 threads,
14 Go), Gateway Open + Ollama (qwen2.5:1.5b-instruct, granite-embedding:278m) + TEI
(bge-reranker-v2-m3) ; les deux chemins interrogent la même Gateway.

| Opération | Direct | via Pool | Observation |
|---|---|---|---|
| `GET /health/runtime` | ~1 ms | ~2–3 ms | corps identique (hors `inflight`) |
| `POST /v1/generate/stream` — premier événement | 275 ms | 222 ms | ordre de grandeur identique (bruit CPU), 13 `delta` + 1 `done` des deux côtés |
| `POST /v1/embeddings` (2 textes) | — | — | vecteurs **identiques** (différence maximale 0,0) |
| `POST /v1/rerank` (4 documents) | — | — | scores et ordre identiques |

Les latences d'inférence sont dominées par le CPU (voir le dossier de qualification interne du programme) ;
le Pool n'ajoute qu'un relais en mémoire par morceaux de 64 Kio.

## Dette connue

Le débit en forte concurrence (nombreux flux simultanés relayés par un seul processus) n'a pas été
remesuré dans l'édition Open ; les scripts `scripts/bench_proxy.py` et `scripts/bench_pool3.py`
(faux Gateway, boucle locale) permettent de le mesurer sans GPU.

## Rejouer

```bash
docker compose -f deploy/compose/qualification/compose.yml up -d   # Gateway Open + modèles approvisionnés + Redis + Pool
python3 tests/parity/parity_direct_pool.py --direct http://127.0.0.1:18100 --pool http://127.0.0.1:18090
```
