# Registre des nœuds

## Modèle

* `NodeDescriptor` (statique, configuration) : `node_id`, `physical_host_id`, `display_name`,
  `environment` = `gateway_instance`, `base_url` (`http(s)://hôte[:port]` sans identifiants ni
  chemin), `enabled`, `weight`, `capabilities`, `models` (identité par modèle : `revision`,
  `dimension`, `pipeline_id`), `default_models`, `max_inflight`, `metadata`.
* `NodeState` (observé) : santé, joignabilité, `build_git_sha`, états de capacités, identifiants de
  modèles observés, capacités brutes et digests publiés, défauts observés, requêtes en vol.

## Déclaré ≠ disponible

Une capacité déclarée reste `DECLARED` jusqu'à ce qu'une sonde l'observe `AVAILABLE` ; un état
`UNAVAILABLE` (capacité `DISABLED`/`UNAVAILABLE` publiée par la Gateway Open, ou nœud d'un autre
environnement) exclut le nœud du routage pour cette capacité.

## Identité des modèles

`model_id` est l'identifiant exact accepté par la Gateway : nom **logique** du registre Compact
(`compact-generation-8`, `fr-en-retrieval`, `rerank-default`) ou tag amont (`qwen2.5:1.5b-instruct`,
`granite-embedding:278m`). Pour les embeddings, `revision` (digest) + `dimension` + `pipeline_id`
épinglent l'espace vectoriel ; deux modèles d'embeddings ne sont jamais interchangeables parce
qu'ils partagent une dimension.

## Règles de validation

Identifiants `^[a-z0-9][a-z0-9-]{0,62}$` ; `node_id` et `base_url` uniques ; environnement du nœud =
environnement du fichier ; capacités déclarées ⊆ capacités du contrat Open (StructX refusé) ;
capacités des modèles ⊆ capacités du nœud ; modèle par défaut déclaré avec sa capacité ; politique
`SINGLE_NODE` ⇒ exactement un nœud activé ; `upstream_allowlist` couvrant chaque URL (obligatoire en
configuration approuvée) ; chemin de sonde ∈ {`/health/runtime`}.

## Sécurité

Le registre est la **seule** source de destinations ; `resolve_destination` ne connaît que les nœuds
configurés. Aucune URL n'est construite depuis une requête.

## Découverte bornée

`probe_on_startup` / `background_probing` (intervalle ≥ 10 s), deux GET avec le profil de délai
`health_probe` (connect 2 s, read 3 s, budget 5 s). Toute réponse HTTP prouve la joignabilité et
permet à un disjoncteur HALF_OPEN de se refermer sans trafic client.

## Ajouter un nœud

Déclarer le nœud (même environnement), ses capacités et modèles avec leur identité, l'ajouter à
`upstream_allowlist`, valider (`ai-gateway-pool-config validate`), redéployer. Un nœud dont le
digest observé diverge de la révision déclarée n'est jamais utilisé.
