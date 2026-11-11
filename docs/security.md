# Sécurité

## Principes appliqués

* Le Pool n'est **pas un proxy HTTP ouvert** : la destination vient exclusivement du registre de
  nœuds (configuration validée) ; un client ne peut ni fournir une URL (`UPSTREAM_OVERRIDE_DENIED`
  sur `X-Upstream-Url`, `X-Forwarded-Host`, `Forwarded`, …), ni atteindre une route hors contrat
  (`ROUTE_NOT_IN_CONTRACT`), ni cibler un nœud sans le jeton d'administration
  (`TARGET_NODE_FORBIDDEN`).
* **Allowlist d'upstreams** (`upstream_allowlist`, obligatoire en configuration approuvée) : hôtes
  exacts, IP, CIDR ou `hôte:port` ; une URL de nœud hors liste est refusée au chargement.
* **Aucune substitution de modèle** : `model_substitution: FORBIDDEN` est la seule valeur ; le modèle
  demandé est servi exactement ou refusé (`MODEL_NOT_AVAILABLE`), jamais remplacé.
* **Fail-closed** : configuration invalide ⇒ pas de démarrage ; corps JSON non inspectable ⇒ refus ;
  magasin d'affinité indisponible ⇒ refus avant toute création de job ; nœud d'un autre environnement
  ⇒ `UNAVAILABLE`.

## Surface HTTP

| Routes | Exposition | Protection |
|---|---|---|
| `/health/live`, `/pool/health/*` | réseau interne / boucle locale | aucune donnée sensible (ni URL, ni IP, ni modèle) |
| `/v1/*`, `/health/*` (contrat Gateway) | réseau interne | catalogue fermé, inspection du modèle, allowlist, disjoncteur |
| `/internal/*` | réseau interne | jeton `POOL_ADMIN_TOKEN` (`Authorization: Bearer`) ; routes désactivées (404) si aucun jeton n'est configuré |

Le conteneur n'écoute que dans son espace réseau Compose ; seul l'exploitant publie le port du Pool,
en boucle locale (`127.0.0.1:18090` dans les déploiements fournis). Gateway, Ollama et TEI restent sur
leur réseau interne.

## Durcissement AIGP-HARDEN-1 (préservé et testé lors de la qualification)

1. Validation stricte de la configuration (pydantic `extra=forbid`, bornes, cohérence environnement/nœuds/modèles).
2. Allowlist des upstreams, épinglage du port.
3. Rejet des en-têtes de redirection d'upstream et du cadrage ambigu (`Content-Length` + `Transfer-Encoding`).
4. Inspection bornée du corps JSON avant tout relais ; `model` dupliqué, non textuel ou trop long refusé.
5. Routage par capacités déclarées **et observées** ; identité des modèles vérifiée contre le digest publié.
6. Affinité des jobs ASR (nœud propriétaire), jamais de diffusion à tous les nœuds.
7. Disjoncteur sur pannes de transport, états CLOSED / OPEN / HALF_OPEN, essai HALF_OPEN nettoyé.
8. Saturation : `503 + Retry-After` d'un nœud ⇒ suspension bornée ; admission par `max_inflight`.
9. Timeouts bornés par classe ; aucun retry de POST ; une seule relance d'un GET sans octet reçu.
10. Déconnexion client propagée à l'upstream (flux annulé).
11. `X-Request-ID` canonique (motif borné) sur toute réponse.
12. `X-AIGP-*` réservé : jamais accepté d'un client, jamais relayé depuis un upstream.
13. Erreurs Pool typées et marquées, distinctes des erreurs Gateway relayées.
14. Healthchecks non inférentiels uniquement.

Tests : `tests/unit/test_harden1_breaker.py`, `tests/integration/test_harden1_*`,
`tests/integration/test_proxy_security_models.py`, `tests/contracts/`, `tests/redis/`.

## Multi-nœuds et Redis

L'affinité de jobs partagée entre processus repose sur Redis (`job_affinity.backend: redis`) ; l'URL
est lue dans la variable d'environnement **nommée** par `redis_url_env`, jamais écrite dans le YAML.
Clés préfixées par environnement (`aigp:<env>:asr-job:<id>`) : aucune lecture croisée staging /
production. Redis doit exiger un mot de passe (`--requirepass`) même sur réseau interne.

## Client HTTP sortant

Un client partagé, timeouts par classe, keep-alive sous la valeur du serveur, aucun suivi de
redirection, destination = URL du nœud + chemin brut validé.

## Authentification

Le Pool ne porte pas d'authentification utilisateur : la confiance est réseau (réseaux internes,
aucun port public), comme pour AI Gateway Open. Un `Authorization` client est relayé à la
Gateway sans être journalisé ; le jeton d'administration du Pool, lui, n'est **jamais** relayé.

## Journalisation

Journal JSON structuré : identifiants, statuts, latences, nœud, capacité, `model_id`, catégorie
d'erreur. **Jamais** de corps de requête, de prompt, de document ni de jeton (test de non-fuite
`tests/unit/test_transport_observability.py`).

## Secrets

Aucun secret dans l'image, le dépôt, les configurations ou les exemples. `POOL_ADMIN_TOKEN` et le mot
de passe Redis sont générés localement (`openssl rand -hex 32`) et passés par l'environnement
(`.env` en 0600, ignoré par git). `scripts/check-secrets.sh` refuse toute valeur de secret dans un
fichier d'exemple ; le programme OPEN rejoue ses propres scanners avant toute publication.

## Conteneur

Image sans configuration, utilisateur non privilégié (uid 10001), `HEALTHCHECK` sur `/health/live`,
arrêt gracieux par SIGTERM, dépendances installées depuis un lock à hashes.
