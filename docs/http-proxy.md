# Relais HTTP

## Statut

Relais transparent des 19 routes de `gateway-api/1.0` (plus les routes propres au Pool :
`/health/live`, `/pool/health/*`, `/internal/*`). Qualifié en mono-nœud contre une AI Gateway
Open réelle ; multi-nœuds exercé avec des nœuds simulés.

## Démarrer localement

```bash
export POOL_ENVIRONMENT=production POOL_CONFIG_FILE=$PWD/config/compact.example.yaml
.venv/bin/ai-gateway-pool                      # 127.0.0.1:8080 -> http://ai-gateway:8100
curl -s http://127.0.0.1:8080/health/live       # {"status":"live"}
curl -s http://127.0.0.1:8080/pool/health/ready # état du Pool (nœuds, affinité)
```

Sans Gateway joignable (ex. `config/local.example.yaml`), les requêtes `/v1/*` répondent
`503 UPSTREAM_UNAVAILABLE` : le relais échoue proprement, il n'invente rien.

## Cheminement d'une requête

1. Middleware : identifiant de requête (`X-Request-ID` relayé si valide, sinon généré), refus du
   cadrage ambigu (`Content-Length` + `Transfer-Encoding`), refus des en-têtes de redirection
   d'upstream (`UPSTREAM_OVERRIDE_DENIED`), journal d'accès.
2. Catalogue : la route doit exister dans le contrat (`ROUTE_NOT_IN_CONTRACT` sinon) ; chemin brut
   validé (`INVALID_REQUEST_PATH`).
3. Corps JSON des routes à modèle : lu **en entier et borné** (`transport.json_inspect_max_bytes`),
   `model` établi sans ambiguïté (`INVALID_REQUEST_BODY`, `REQUEST_BODY_TOO_LARGE`,
   `UNSUPPORTED_CONTENT_ENCODING`, `INSPECTION_BUDGET_EXHAUSTED`).
4. Sélection : nœud activé, environnement conforme, capacité déclarée **et observée**, modèle
   déclaré **et observé** (identité vérifiée contre le digest publié), disjoncteur fermé, capacité
   en vol disponible ; sinon erreur typée (`MODEL_NOT_AVAILABLE`, `CAPABILITY_NOT_AVAILABLE`,
   `NODE_UNAVAILABLE`, `NO_ELIGIBLE_NODE`).
5. Envoi : corps relayé octet pour octet ; réponse relayée (statut, en-têtes filtrés, corps ou
   flux) ; `Retry-After` 503 enregistré comme saturation du nœud.

## En-têtes

| Sens | Retirés | Conservés |
|---|---|---|
| client → Gateway | hop-by-hop, `Host` (recalculé), `X-Upstream-*`/`Forwarded`/… (refusés), `X-AIGP-*`, `Authorization` portant le jeton admin du Pool | tout le reste, y compris `Content-Length`, `Accept`, un `Authorization` client |
| Gateway → client | hop-by-hop, `Server`, tout `X-AIGP-*` (jamais forgeable par un upstream) | statut, `Content-Type`, `Retry-After`, `Date`, corps |

Sur chaque réponse : `X-Request-ID` ; sur chaque refus du Pool : `X-AIGP-Error: <code>` et corps
`{"error": {"code", "message", "request_id"}}`.

## Erreurs

Les codes du Pool (`src/ai_gateway_pool/errors.py`) sont distincts des erreurs Gateway : un client
reconnaît un refus du Pool à `X-AIGP-Error` et à la forme `{"error": ...}` ; une erreur Gateway
garde sa forme `{"detail": ...}` et son statut. Exemples qualifiés : `MODEL_NOT_AVAILABLE` (404),
`CAPABILITY_NOT_AVAILABLE` (503), `ROUTE_NOT_IN_CONTRACT` (404), `UPSTREAM_OVERRIDE_DENIED` (400),
`TARGET_NODE_FORBIDDEN` (403), `INVALID_REQUEST_BODY` (400), `UPSTREAM_UNAVAILABLE` (503),
`UPSTREAM_TIMEOUT` (504), `ROUTING_NOT_READY` (501).

## Timeouts

Configurables par classe (`timeouts:` — `health_probe`, `inference`, `streaming`, `upload`,
`job_poll`) ; un flux n'a pas de budget total, seulement un délai entre fragments.

## Basculement contrôlé

Uniquement lorsque la requête n'a **provablement** jamais quitté le Pool (connexion refusée, délai
de connexion, aucune connexion disponible), vers un autre nœud compatible du même environnement,
et jamais pour les routes de diagnostic ni les routes de job épinglées. Mono-nœud : désactivé.

## Retries

Aucun POST d'inférence n'est rejoué (un travail GPU a pu commencer). Un GET idempotent est rejoué
une fois, même nœud, uniquement si aucun octet de réponse n'a été reçu. `DELETE` n'est jamais rejoué.

## Connexions

Un client HTTP partagé par processus (`transport.max_connections`, `max_keepalive_connections`) ;
expiration keep-alive 4 s, sous les 5 s d'uvicorn côté Gateway, pour ne jamais réutiliser une socket
que le serveur va fermer.

## Sonde de santé

`GET /health/live` (Pool, Docker HEALTHCHECK) et `GET /pool/health/ready` (nœuds utilisables,
magasin d'affinité). Le Pool ne sonde ses nœuds que par `/health/runtime` et `/v1/models`, tous deux
non inférentiels ; `/health/ready`, `/health/embeddings` et `/health/rerank` sont **refusés** comme
chemins de sonde par la validation de configuration.
