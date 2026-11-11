# Contrat AI Gateway Open relayé — `gateway-api/1.0`

Source de vérité : `gateway-api/1.0.yaml` publié par AI Gateway Open (OpenAPI 3.1, 19 routes,
statut `CANDIDATE_INTEGRATION_VERIFIED`, ratification en attente). Le Pool ne modifie ni les
schémas, ni les en-têtes, ni les événements de flux. Snapshot de test : `tests/fixtures/gateway/openapi-routes.json`.

## Routes (20 dans le snapshot : 19 relayées + 1 locale au Pool)

| Méthode | Route | Capacité | Corps | Réponse | Inférentielle | Comportement Pool |
|---|---|---|---|---|---|---|
| GET | `/health/live` | — | — | JSON | non | **POOL_LOCAL** : liveness du Pool, jamais relayée |
| GET | `/health` | — | — | JSON | non (`/api/tags`) | relayée |
| GET | `/health/runtime` | — | — | JSON | non | relayée ; **seule sonde** utilisée par le Pool |
| GET | `/health/ready` | EMBEDDINGS | — | JSON | **oui** | relayée ; jamais sondée par le Pool |
| GET | `/health/embeddings` | EMBEDDINGS | — | JSON | **oui** | relayée ; jamais sondée |
| GET | `/health/rerank` | RERANK | — | JSON | **oui** | relayée ; jamais sondée |
| GET | `/health/vision` | VISION | — | JSON | non | relayée (capacité `DISABLED` en Compact) |
| GET | `/health/asr` | ASR | — | JSON | non | relayée (capacité `DISABLED` en Compact) |
| GET | `/v1/models` | — | — | JSON | non (inventaire) | relayée telle quelle (modèles **installés** seulement) |
| GET | `/v1/vision/capacity` | VISION | — | JSON | non | relayée |
| POST | `/v1/generate` | GENERATE | JSON (`model` inspecté) | JSON | oui | relayée après contrôle du modèle |
| POST | `/v1/generate/stream` | GENERATE | JSON | NDJSON | oui | relayée en flux, octet pour octet |
| POST | `/v1/chat` | CHAT | JSON | JSON | oui | relayée (aucun modèle Compact ne déclare `chat`) |
| POST | `/v1/chat/stream` | CHAT | JSON | NDJSON | oui | relayée en flux |
| POST | `/v1/embeddings` | EMBEDDINGS | JSON | JSON | oui | relayée ; vecteurs jamais transformés |
| POST | `/v1/rerank` | RERANK | JSON | JSON | oui | relayée ; scores et ordre jamais modifiés |
| POST | `/v1/vision/analyze` | VISION | JSON | JSON | oui | relayée si la capacité est déclarée |
| POST | `/v1/audio/transcriptions` | ASR | multipart | JSON | oui | relayée si affinité de jobs configurée |
| GET | `/v1/audio/transcriptions/{job_id}` | ASR | — | job | non | épinglée au nœud propriétaire du job |
| DELETE | `/v1/audio/transcriptions/{job_id}` | ASR | — | JSON | non | épinglée ; jamais relancée |

Absentes de l'édition Open (refusées `404 ROUTE_NOT_IN_CONTRACT`, jamais dispatchées) :
`POST /v1/vision/structured-extract`, `GET /health/structured-extract`.

## Ce que le relais préserve

* **Corps** relayé sans réécriture (le `model` d'un corps JSON est lu pour la gouvernance, puis le
  corps part tel quel) ; `Content-Length` conservé ; `Content-Encoding` refusé sur les routes JSON.
* **Flux NDJSON** (`application/x-ndjson`, événements `delta` / `done` / `error`) relayé par
  morceaux sans agrégation ni reformulation : exactement un `done` par flux réussi, jamais de `done`
  après `error`, annulation cliente propagée à l'upstream.
* **Identité des modèles** : `model`, `model_logical`, `model_digest`, `expected_digest`,
  `identity_verified`, `pipeline_id`, `model_revision` passent sans altération ; le Pool n'impose
  aucun alias — `nomic-embed-text` ≡ `nomic-embed-text:latest` est la seule normalisation
  (référence Ollama), appliquée au **contrôle**, jamais au corps.
* **Erreurs Gateway** (`{"detail": {status, provider, message, code?, retryable?}}`, statut HTTP,
  `Retry-After`) relayées sans modification ; une saturation `503` + `Retry-After` suspend le nœud
  pour la durée annoncée (bornée par `circuit_breaker.saturation_max_seconds`).
* **En-têtes** : `X-Request-ID` relayé s'il est valide, sinon généré ; hop-by-hop et en-têtes de
  redirection d'upstream retirés ; `X-AIGP-*` réservé au Pool (un `X-AIGP-Error` venant du client
  ou de la Gateway n'est jamais relayé).

## Codes d'erreur Gateway observés en qualification (Gateway Open réelle)

| Code | HTTP | Relayé par le Pool ? |
|---|---|---|
| `model_not_registered` | 400 | relayé — mais en pratique le Pool refuse d'abord (`MODEL_NOT_AVAILABLE`, 404) tout modèle absent de son allowlist |
| `model_not_installed` (`retryable: false`) | 503 | relayé tel quel si le modèle est déclaré ; **jamais réessayé** par le Pool (POST non rejoués) |
| `generation_saturated`, `embeddings_saturated`, `rerank_saturated` | 503 + `Retry-After` | relayés ; nœud suspendu le temps annoncé |
| `length_required`, `request_too_large`, `invalid_content_length` | 411 / 413 / 400 | relayés |
| `embedding_dimension_incompatible` | 400 | relayé |
| `extra_forbidden`, bornes (`max_input_chars`, lots, `num_ctx`) | 400 | relayés |

## Délais (profil Compact)

| Classe | Pool (`config/compact.example.yaml`) | Gateway Open | RAGatm Open |
|---|---|---|---|
| sonde | connect 2 s, read 3 s, budget 5 s | — | — |
| inférence | read 330 s | `AI_GATEWAY_TIMEOUT_SECONDS=300` | 300 s |
| flux | read 330 s entre fragments | 300 s | 300 s |
| rerank | (inférence) | 30 s | 20 s |

Le Pool attend plus longtemps que ses clients : il ne coupe jamais une réponse que le client
attend encore, et ne retient pas une connexion que le client a abandonnée (annulation propagée).
