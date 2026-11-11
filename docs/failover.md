# Basculement contrôlé (POOL-3)

Code : `GatewayRelay._forward` / `_send_once` et `failover_allowed()` dans
[src/ai_gateway_pool/transport/proxy.py](../src/ai_gateway_pool/transport/proxy.py).
Matrice des 21 opérations : `docs/audits/POOL-3/route-failover-matrix.csv`.

## Principe

> Un second nœud disponible ne rend pas une requête rejouable.

Le pool ne choisit un autre nœud **que** si la requête n'a, de façon prouvée, jamais
quitté le pool :

| Échec | Preuve que rien n'a été envoyé | Bascule |
|---|---|---|
| `httpx.ConnectError` (connexion refusée / impossible) | oui : levée pendant l'établissement de la connexion TCP, avant toute écriture | **oui** (routes éligibles) |
| `httpx.ConnectTimeout` | oui | **oui** |
| `httpx.PoolTimeout` (pas de connexion libre dans le pool local) | oui | **oui** |
| corps de requête en flux déjà entamé | non | jamais (`TrackedBody.started`) |
| `RemoteProtocolError` (connexion perdue), `ReadError`, `WriteError` | **non** — la Gateway a pu recevoir et exécuter | jamais |
| `ReadTimeout` / budget dépassé avant les en-têtes | non — calcul possiblement en cours | jamais |
| réponse amont reçue (y compris `503`) | — | jamais : relayée telle quelle |
| flux de réponse commencé puis interrompu | — | jamais : connexion aval interrompue, pas de `done` inventé |

Une connexion keep-alive réutilisée qui casse produit `RemoteProtocolError`/`ReadError`
(pas `ConnectError`) : elle est donc classée « résultat incertain », jamais rejouée pour un
POST. Un `GET` sans corps garde la règle POOL-2 : un rejeu, **même nœud**, si aucun octet
de réponse n'a été reçu.

## Règles par route

| Classe | Routes | Bascule |
|---|---|---|
| Diagnostic / inventaire | 8 × `/health*`, `GET /v1/models`, `GET /v1/vision/capacity` | **non** : la réponse décrit UN nœud (identité, `build.git_sha`, environnement) |
| Inférence POST | generate, generate/stream, chat, chat/stream, embeddings, rerank, vision/analyze, vision/structured-extract | **avant envoi uniquement** |
| Création de job ASR | `POST /v1/audio/transcriptions` | **avant envoi uniquement** (le job sera lié au nœud qui l'accepte) |
| Job ASR | `GET` / `DELETE /v1/audio/transcriptions/{job_id}` | **non** : nœud imposé par l'affinité, aucun nœud exploré |

La nouvelle sélection passe par le même sélecteur, avec `exclude_nodes` = nœuds déjà
tentés : environnement, capacité, modèle exact, identité d'artefact et santé sont
réévalués. Un nœud incompatible n'est **jamais** un recours de dernier ressort.
`failover.max_extra_nodes` (défaut 1) borne le nombre de nœuds supplémentaires.

## Sélection préalable (avant toute tentative)

Avant l'envoi, le sélecteur écarte : nœud désactivé, sonde `UNAVAILABLE`, disjoncteur
`OPEN` (ou `HALF_OPEN` avec son essai déjà en cours), `max_inflight` atteint. Un nœud
qui a répondu récemment `503` + `Retry-After` est écarté **seulement s'il existe un autre
candidat** (exclusion douce, bornée par `circuit_breaker.saturation_max_seconds`).

## Disjoncteur (local au processus)

`CLOSED → OPEN` après `failure_threshold` échecs de **transport** consécutifs ;
`OPEN → HALF_OPEN` après `open_seconds` ; un seul essai ; succès → `CLOSED`, échec → `OPEN`
avec durée doublée (plafond `max_open_seconds`). Toute réponse HTTP (4xx, `503
resource_saturated`, 5xx) prouve que le nœud répond : elle referme le disjoncteur et ne
compte jamais comme panne. Aucune sonde inférentielle n'est utilisée.

**Essai sans verdict (AIGP-HARDEN-1).** Si l'essai `HALF_OPEN` se termine sans réponse
HTTP **ni** échec de transport — client parti (`499`), corps client tronqué
(`INVALID_REQUEST_FRAMING`), pas de connexion libre dans le pool local
(`UPSTREAM_TIMEOUT`), annulation, erreur inattendue — l'essai est **libéré** : le
disjoncteur reste `HALF_OPEN` et la requête suivante peut retenter. Une annulation
client ne condamne donc jamais le nœud, et aucune séquence n'exige un redémarrage du
pool. Une requête démarrée avant l'ouverture ne peut pas libérer l'essai d'une autre
(un seul essai à la fois).

**Sonde.** Toute réponse HTTP de la sonde `/health/runtime` (fond ou
`POST /internal/nodes/{id}/probe`) referme un disjoncteur `HALF_OPEN` sans essai en
cours : un nœud rétabli redevient utilisable même sans trafic client. Une sonde
n'abrège jamais une période `OPEN` (le backoff est conservé) et ne tranche pas à la
place d'un essai en cours.

## Requête épinglée

Une requête épinglée — `X-AIGP-Target-Node` (réservé au jeton admin du pool) ou
affinité d'un job ASR — n'est **jamais** basculée vers un autre nœud, même en cas
d'échec avant envoi : le client reçoit l'erreur du nœud qu'il a choisi.

## Erreurs stables

* aucun nœud joignable / plus d'alternative compatible : `503 UPSTREAM_UNAVAILABLE`
  (après tentative) ou `503 NODE_UNAVAILABLE` (avant envoi) ;
* aucune bascule entre environnements : impossible par construction (le registre d'un
  processus ne contient qu'un environnement).

## Observabilité

Champs de journal `failover_attempted`, `failover_allowed`, `failover_result`
(`success`, `no_alternative`, `exhausted`) ; métriques
`aigp_pool_failover_attempts_total`, `aigp_pool_failover_success_total`,
`aigp_pool_failover_rejected_total{error_category}`.
