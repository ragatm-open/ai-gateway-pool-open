# Politiques de routage

| Politique | État POOL-1 |
|---|---|
| `SINGLE_NODE` | **implémentée et testée** — politique V1 |
| `WEIGHTED_ROUND_ROBIN` | implémentée et qualifiée sur 2–3 faux nœuds HTTP (POOL-3) ; pas de trafic multi-GPU réel |
| `TARGETED_NODE` | option **par requête** (`target_node_id`) respectée par tout sélecteur, pas une politique de pool |
| `LEAST_INFLIGHT` | **POOL-3** : requêtes en cours de ce processus, agrégées par hôte physique |
| `LEAST_BUSY` | non implémentée (télémétrie GPU absente) → config refusée `ROUTING_NOT_READY` |
| `CAPABILITY_AWARE` | pas une politique distincte : le filtrage capacité/modèle/identité s'applique à **toutes** les politiques (POOL-3) ; nom refusé en config |

`LEAST_BUSY` exigera un signal de charge fiable. `/health/runtime.inflight` est
local à l'instance et reste bloqué après une déconnexion de flux (défaut connu
de la Gateway) : il ne suffit pas.

## Éligibilité d'un nœud

1. environnement de la requête = environnement du pool (sinon `NO_ELIGIBLE_NODE`) ;
2. nœud activé ;
3. capacité déclarée (sinon `CAPABILITY_NOT_AVAILABLE`) ;
4. modèle **exactement** servi (sinon `MODEL_NOT_AVAILABLE`) ; si l'inventaire a
   été observé, le modèle doit y figurer ;
5. nœud non observé `UNAVAILABLE` (sinon `NODE_UNAVAILABLE`) ;
6. capacité non observée `UNAVAILABLE` (sinon `CAPABILITY_NOT_AVAILABLE`).

Aucun repli hors de l'ensemble éligible. Un nœud ciblé indisponible n'est pas
remplacé par un autre.

## Embeddings : aucune substitution

`model_substitution` n'accepte que `FORBIDDEN`. `models_compatible` exige l'égalité
de `model_id`, puis — pour les embeddings — de la `revision` et de la `dimension`
si elles sont épinglées. **Une dimension égale ne rend jamais deux modèles
compatibles** : `legacy-768` et `fr-en-retrieval` font tous deux 768 dimensions
et ne sont pas interchangeables (test
`test_model_compatibility_required_same_dimension_is_not_enough`).

## Erreurs

| Code | HTTP (pool) | Cas |
|---|---|---|
| `NO_ELIGIBLE_NODE` | 503 | aucun nœud utilisable / environnement étranger |
| `NODE_UNAVAILABLE` | 503 | nœuds compatibles tous indisponibles |
| `MODEL_NOT_AVAILABLE` | 404 | modèle non servi (substitution interdite) |
| `CAPABILITY_NOT_AVAILABLE` | 503 | capacité non déclarée ou observée indisponible |
| `ROUTING_NOT_READY` | 501 | politique ou relais non implémenté |
| `CONFIGURATION_INVALID` | 500 | configuration incohérente |
| `INVALID_NODE_DESTINATION` | 400 | nœud ciblé inconnu ou désactivé |
| `UPSTREAM_OVERRIDE_DENIED` | 400 | en-tête tentant d'imposer une destination |
| `ROUTE_NOT_IN_CONTRACT` | 404 | route `/v1/*` absente du contrat |

Corps : `{"error": {"code", "message", "request_id"}}`, en-tête `X-AIGP-Error`.
Les statuts et corps amont sont relayés inchangés (POOL-2) ; codes de transport et
d'affinité : [http-proxy.md](http-proxy.md#erreurs).

## Routes de diagnostic

`/health*`, `/v1/models`, `/v1/vision/capacity` : `RoutingRequest.capability = None`.
Pas de contrôle de capacité ni de modèle, et relais même vers un nœud observé
`UNAVAILABLE` : l'opérateur reçoit la vraie réponse du nœud (ou `UPSTREAM_UNAVAILABLE`).
Jamais vers un nœud d'un autre environnement.

## Identités de modèles

Le pool lit `model` dans les corps JSON (≤ 16 Mio) ; sans `model`, routage par capacité.
Identités déclarées : logiques (`model_logical`, capacités de `/health/runtime`) **et**
concrètes (`/v1/models`, ex. `llama3.1:8b` envoyé par RAGatm). Elles ne sont jamais
traitées comme alias : chaque identifiant est relayé tel quel. Après une sonde, le modèle
doit aussi figurer dans l'inventaire observé.

## Retries

Jamais de rejeu automatique d'un POST d'inférence (la Gateway a pu commencer le
calcul ; l'ASR n'a pas de clé d'idempotence). Seules les lectures idempotentes
(GET/HEAD/OPTIONS) peuvent être rejouées une fois, **sur le même nœud**.

## POOL-3

Détail de l'évaluation multi-nœuds, des raisons d'exclusion, du modèle implicite et de
l'identité des artefacts : [multi-node.md](multi-node.md). Basculement : [failover.md](failover.md).
Nouveau code d'erreur : `MODEL_REQUIRED` (400) — requête sans `model` alors que les nœuds
candidats ne publient pas un défaut identique.
