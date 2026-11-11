# Architecture

## Rôle des composants

```text
AI GATEWAY POOL OPEN = routage et orchestration entre nœuds AI Gateway Open (aucun GPU, aucune inférence)
AI GATEWAY OPEN      = exécution IA locale (Ollama, TEI) et gouvernance des modèles, par machine
RAGATM OPEN          = ingestion documentaire, recherche hybride, génération avec citations
```

Le Pool est **facultatif** : RAGatm Open consomme le même contrat `gateway-api/1.0` en accès
direct (`http://ai-gateway:8100`) ou via le Pool (`http://pool:8080`). Le changement de chemin est
une décision de configuration du client, jamais un changement de logique applicative.

## Topologie de référence

| Rôle | Exemple (noms de services Compose, pas de machines) |
|---|---|
| Poste de développement | lance les tests, construit l'image du Pool (`scripts/check-local.sh`) |
| Hôte du Pool | conteneur `pool` (+ `redis` si affinité de jobs), sur un réseau interne |
| Nœud(s) d'exécution IA | une ou plusieurs AI Gateway Open (`ai-gateway`, ou `gateway-a`, `gateway-b`…), chacune devant son Ollama et son TEI |

Un seul nœud est **qualifié** (Gateway Open réelle). Plusieurs nœuds sont **préparés**
(nœuds simulés + Redis réel) mais `MULTINODE_HARDWARE_QUALIFIED = NO`.

## Découpage interne

```text
            HTTP (FastAPI)
  ┌──────────────┬────────────────────┬─────────────────────────┐
  │ /health/live │ /internal/* (jeton) │ /v1/* + /health/* relayés (contrat Gateway) │
  │ /pool/health │                    │                                            │
  └──────┬───────┴─────────┬──────────┴───────────┬─────────────────────────────────┘
         │                 │                      │ relais (transport/proxy.py)
         ▼                 ▼                      ▼ flux NDJSON, affinité ASR, métriques
   NodeRegistry ◄── discovery (2 GET bornés) ── NodeSelector (routing/)
     (nodes/)            │                        │
         ▲               ▼                        ▼
   PoolConfig      gateway/health.py       transport/contracts.py
   (config/)       (interprétation)        (timeouts, retries, en-têtes)
                                           jobs/ (affinité job → nœud : memory | sqlite | redis)
```

* `config/` valide **un** fichier par environnement ; `NodeRegistry` en est la **seule** source de
  destinations (un client ne choisit jamais une URL).
* `gateway/contracts.py` est le catalogue des routes relayées : **19 routes** de `gateway-api/1.0`
  (+ `GET /health/live`, servie par le Pool lui-même). Les deux routes historiques exclues de
  l'édition Open (StructX) ne sont pas dispatchables (`ROUTE_NOT_IN_CONTRACT`).
* `routing/` ne connaît pas HTTP : il reçoit un `RoutingRequest` (environnement, service, capacité,
  modèle, nœud ciblé éventuel) et renvoie une décision ou une erreur typée.
* `transport/` relaie octet pour octet, sans transformer le NDJSON, sans réécrire les corps.

## Nœuds physiques et logiques

`node_id` identifie une instance de Gateway ; `physical_host_id` identifie la machine (le GPU).
Deux Gateways d'une même machine (par exemple une instance staging et une instance production)
partagent le même GPU et ne comptent jamais pour deux : les politiques agrègent par hôte physique.

## Choix structurants

1. Un fichier de configuration = un environnement ; un nœud d'un autre environnement est refusé.
2. Déclaré ≠ disponible : une capacité ou un modèle déclaré n'est routé qu'après observation par
   une sonde **non inférentielle** (`/health/runtime`, `/v1/models`).
3. Aucune substitution de modèle : le `model` demandé est servi exactement ou la requête est refusée.
4. Les erreurs du Pool portent `X-AIGP-Error` et un corps `{"error": {...}}` ; celles de la Gateway
   (`{"detail": {...}}`) sont relayées sans modification.
5. Aucun secret dans l'image ni dans les fichiers versionnés ; jeton d'administration et mot de passe
   Redis fournis par l'environnement.
