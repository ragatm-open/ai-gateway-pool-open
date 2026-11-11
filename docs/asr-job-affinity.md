# Affinité des jobs ASR

**Profil Compact Open : ASR désactivée** (`AI_GATEWAY_ASR_ENABLED=0`, aucun backend inclus). Dans
`config/compact.example.yaml`, `job_affinity.backend: none` : les routes `/v1/audio/*` sont refusées par
le Pool (`501 ROUTING_NOT_READY`) sans contacter la Gateway, qui les refuserait elle-même
(`503 asr_unavailable`). Cette page décrit le mécanisme conservé pour les installations qui
fournissent un backend ASR qualifié.

## Problème

`POST /v1/audio/transcriptions` (mode `async`) crée un job **local à une instance** de Gateway : le
`GET` et le `DELETE /v1/audio/transcriptions/{job_id}` doivent atteindre le **même nœud**. Aucune clé
d'idempotence n'existe côté Gateway : un rejeu créerait un second job.

## Contrat `JobAffinityStore` (`jobs/contracts.py`)

| Opération | Garantie |
|---|---|
| `register_job(env, job_id, node_id, ttl)` | crée l'association ; refuse (`JOB_AFFINITY_CONFLICT`) d'écraser une association vivante vers un autre nœud ; idempotent vers le même nœud |
| `lookup_job(env, job_id)` | association ou `None` ; jamais de lecture croisée entre environnements |
| `mark_job_completed(env, job_id, ttl)` | statut terminal, durée de vie prolongée pour la relecture |
| `forget_job(env, job_id)` | suppression après un `DELETE` amont réussi |
| `ping()` | disponibilité du magasin, intégrée à `/pool/health/ready` |

`job_id` conservé tel quel (opaque, borné, sans espace) ; aucune hypothèse d'unicité globale entre
nœuds (collision détectée et refusée) ; mapping absent ⇒ `404 JOB_AFFINITY_UNKNOWN`, jamais de
diffusion à tous les nœuds.

## Implémentations

| Backend | Durable | Usage |
|---|---|---|
| `none` | — | ASR désactivée (Compact) |
| `memory` | non | développement uniquement (refusé hors `development`) |
| `sqlite` | fichier local | développement / qualification ; jamais dans une configuration approuvée |
| `redis` | oui, partagé | cible des déploiements ; URL via variable d'environnement nommée ; clés `aigp:<env>:asr-job:<id>` |

## Comportement du relais

* Soumission : le magasin doit répondre (`ping`) **avant** de créer un job (`AFFINITY_STORE_UNAVAILABLE`
  sinon) ; après création, échec d'écriture de l'association ⇒ compensation (`DELETE` amont du job) et
  erreur explicite — jamais un job orphelin silencieux.
* `GET`/`DELETE` : épinglés au nœud propriétaire ; aucun basculement, aucun rejeu de `DELETE`.
* Redis indisponible : refus fail-closed, reprise automatique au retour (tests `tests/redis/`).

## Qualification

Redis réel (7.2, mot de passe) : 29 tests `redis_real` passés, dont indisponibilité (conteneur mis
en pause) et reprise, multi-processus, collision d'identifiants, isolation staging/production.
Aucun backend ASR réel : chaîne ASR de bout en bout **non exécutée** (hors profil Compact).
