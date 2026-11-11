# Retour arrière

## Retour à l'image précédente du Pool

Le Pool est sans état : redéployer l'image précédente (`POOL_IMAGE_TAG` dans
`deploy/compose/pool/.env`) suffit. La configuration YAML est versionnée avec l'image qui l'a
validée : la revalider (`ai-gateway-pool-config validate`) avant redémarrage.

## Retirer le Pool du chemin d'un client

RAGatm Open consomme le même contrat en direct : repointer `AI_GATEWAY_BASE_URL`,
`AI_GATEWAY_EMBEDDINGS_BASE_URL` et `RAG_RERANKER_BASE_URL` vers la Gateway (`http://ai-gateway:8100`)
et vider `AI_GATEWAY_POOL_HOSTS`. Aucun changement de logique documentaire ; les identités
d'embeddings stockées sont celles de la Gateway, indépendantes du chemin.

## Configuration

Conserver chaque version de `pool.yaml` à côté de l'image qui l'a servie ; un retour arrière
consiste à redéployer le couple (image, configuration).

## Données Redis

Les associations de jobs (`aigp:<env>:asr-job:*`) ne sont pas une source de vérité documentaire ;
elles expirent (`ttl_seconds`). En l'absence d'ASR (Compact), Redis est vide et peut être recréé.

## Migration entre versions de contrat

Le catalogue (`gateway/contracts.py`) porte `CONTRACT_VERSION = "gateway-api/1.0"` ; une nouvelle
version du contrat Gateway impose de régénérer `tests/fixtures/gateway/openapi-routes.json` depuis le
nouveau fichier OpenAPI et de rejouer la parité Direct/Pool avant tout déploiement.
