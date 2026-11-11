# Séparation des environnements

## Principe

Un fichier de configuration décrit **un** environnement (`development`, `staging`, `production`).
`POOL_ENVIRONMENT` doit le confirmer ; un nœud d'un autre environnement, ou une Gateway qui publie un
autre environnement sur `/health/runtime`, est refusé ou marqué `UNAVAILABLE`. Dans l'édition Open,
« production » est l'étiquette que publie AI Gateway Open par défaut (`AI_GATEWAY_ENVIRONMENT`) ;
elle gouverne les allowlists du registre de modèles, pas un engagement d'exploitation.

## Garde-fous (tous testés)

* `environment` et `gateway_instance` égaux pour chaque nœud, égaux à l'environnement du fichier.
* `check-pair` : deux fichiers (ex. multi-nœuds `staging` et Compact `production`) ne partagent ni
  upstream ni `node_id`.
* Clés Redis préfixées par environnement ; un job créé par un Pool `staging` n'est pas résoluble par
  un Pool `production` (test Redis réel).
* `status: example` refusé en déploiement (`POOL_REQUIRE_APPROVED_CONFIG=1`) ; `status: approved`
  impose `upstream_allowlist` et une affinité durable ou désactivée.

## Ressources partagées : à savoir

Deux Gateways sur la même machine partagent le GPU : `physical_host_id` identique, agrégation des
requêtes en vol par hôte. Un Redis d'affinité peut être partagé entre environnements grâce au
préfixe de clé, mais un Redis par environnement reste préférable.

## Limites

L'édition Open ne livre qu'un profil réel (Compact, un nœud). Les paires staging/production sont
validées statiquement sur des exemples fictifs.
