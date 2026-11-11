# Multi-nœuds

**État :** préparé et testé avec des nœuds **simulés** et un Redis **réel** ; configuration
d'exemple `config/multinode.example.yaml` marquée `EXAMPLE_ONLY — MULTINODE_NOT_HARDWARE_QUALIFIED`.
Aucune campagne sur plusieurs serveurs GPU réels n'a eu lieu : le support multi-GPU **n'est pas
qualifié**.

## Ce qui est en place

* Politiques implémentées : `SINGLE_NODE`, `WEIGHTED_ROUND_ROBIN`, `LEAST_INFLIGHT` (requêtes en vol
  de ce processus, agrégées par hôte physique). `LEAST_BUSY`, `CAPABILITY_AWARE`, `TARGETED_NODE`
  sont nommées mais refusées au chargement (`ROUTING_NOT_READY`).
* Sélection par capacité, modèle (identité exacte), observation, disjoncteur, capacité en vol.
* Basculement contrôlé vers un autre nœud compatible, seulement avant tout envoi.
* Affinité de jobs ASR partagée (Redis) entre processus et réplicas.

## Découverte des capacités

Deux GET bornés et non inférentiels par nœud : `/health/runtime` (environnement, capacités
`ENABLED/DISABLED/UNAVAILABLE`, modèles de génération du registre) et `/v1/models` (modèles
installés, digests). Une observation plus ancienne que `capability_ttl_seconds` **restreint** le
routage au dernier ensemble observé mais ne l'élargit jamais.

## Identité des modèles

Le même `model_id` d'embeddings déclaré sur plusieurs nœuds doit porter la même révision, la même
dimension et le même `pipeline_id` : sinon la configuration est refusée (espaces vectoriels
incompatibles). À l'exécution, un digest observé différent de la révision déclarée exclut le nœud
(`MODEL_IDENTITY_MISMATCH`).

## Modèle implicite

Une requête sans `model` est routée vers un nœud dont `default_models[capacité]` est déclaré ; la
Gateway applique alors son propre défaut (identique par construction au profil Compact).

## Explicabilité

Chaque refus de routage nomme la raison par nœud (`MODEL_NOT_DECLARED`, `MODEL_NOT_OBSERVED`,
`CAPABILITY_UNAVAILABLE`, `CIRCUIT_OPEN`, `AT_CAPACITY`, …) ; `/internal/nodes` (jeton) expose
l'état observé sans URL ni IP.

## Configurations

| Fichier | Usage |
|---|---|
| `config/compact.example.yaml` | mono-nœud réel, profil Compact (qualifié) |
| `config/multinode.example.yaml` | exemple fictif à deux nœuds (`gateway-a`, `gateway-b`), affinité Redis |
| `config/local.example.yaml` | développement sans Gateway (nœud injoignable, affinité SQLite) |
