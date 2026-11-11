# Qualification du contrat — niveaux de preuve

| Niveau | Signification | Où |
|---|---|---|
| `contract_stub` | la carte des routes du Pool correspond au snapshot `gateway-api/1.0` | `tests/contracts/test_gateway_contracts.py` |
| MOCK_QUALIFIED | comportement du relais prouvé contre un faux Gateway (routes, flux, erreurs, affinité, basculement) | `tests/contracts/test_route_relay_matrix.py`, `tests/integration/` |
| REDIS_REAL | affinité et pannes Redis contre un Redis réel | `tests/redis/` (marqueur `redis_real`) |
| LIVE_SINGLE_NODE | parité Direct / Pool contre une AI Gateway Open **réelle** et ses modèles locaux (CPU) | `tests/parity/parity_direct_pool.py`, environnement `deploy/compose/qualification/` |
| LIVE_MULTI_GPU | plusieurs serveurs GPU réels | **NOT_EXECUTED** |

Un test qui passe au niveau MOCK ne prouve pas la compatibilité live ; la parité live porte
sur `/health/live`, `/health/runtime`, `/v1/models`, `/health/embeddings`, `/v1/generate`,
`/v1/generate/stream`, `/v1/embeddings` (document et query, vecteurs identiques), `/v1/rerank`,
les erreurs négatives et les en-têtes. Les routes ASR et Vision ne sont pas qualifiées live (capacités
`DISABLED` dans le profil Compact).

## Ce que la parité compare

Statut HTTP, clés du contrat de réponse, identité des modèles (`model`, `model_logical`, digest,
`pipeline_id`, `model_revision`), vecteurs d'embeddings (égalité numérique stricte), scores et ordre
du reranker, séquence des événements NDJSON (un seul `done`), en-têtes `X-Request-ID` /
`X-AIGP-Error` / `Retry-After`, codes d'erreur Gateway relayés. La génération n'étant pas
déterministe, le texte n'est pas comparé.

## Différences attendues (documentées)

* Un modèle hors allowlist est refusé par le Pool (`404 MODEL_NOT_AVAILABLE`, marqué `X-AIGP-Error`)
  avant que la Gateway ne réponde `400 model_not_registered` : l'erreur reste distinguable et aucune
  substitution n'a lieu.
* `GET /health/live` est servie par le Pool (liveness du Pool), non relayée.
* Le Pool ajoute `X-Request-ID` à toute réponse ; la Gateway le fait aussi, le Pool relaie l'identifiant du client.

## Contrôles live sans appel amont

`/health/live`, `/pool/health/ready`, refus d'en-têtes de redirection, refus de routes hors contrat,
refus d'un jeton admin absent (`/internal/*`), neutralisation d'un `X-AIGP-Error` forgé.
