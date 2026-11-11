# Reprise

## Garanties (vérifiées)

* **Nœud injoignable** : disjoncteur par nœud (pannes de transport uniquement), ouverture bornée
  (`open_seconds` → `max_open_seconds`), essai HALF_OPEN unique, refermeture sur n'importe quelle
  réponse HTTP (sonde ou trafic) ; requêtes refusées `503 NODE_UNAVAILABLE` / `UPSTREAM_UNAVAILABLE`
  pendant l'ouverture, jamais mises en file.
* **Saturation** : `503 + Retry-After` de la Gateway ⇒ nœud suspendu pour la durée annoncée (bornée) ;
  l'erreur est relayée au client avec son `Retry-After`.
* **Redis indisponible** (affinité) : refus fail-closed des routes de job avant toute création ;
  reprise automatique au retour (tests `tests/redis/test_redis_reconnect_real.py`,
  `test_multiprocess_pool.py::test_redis_outage_fail_closed_then_recovery`).
* **Redémarrage du Pool** : aucun état requis hors configuration ; les associations de jobs vivent
  dans Redis (durable) et survivent (`test_affinity_survives_process_restart`).
* **Connexion cliente interrompue** : le flux amont est fermé ; aucun second envoi.

## Procédure

1. `GET /pool/health/ready` : nœuds utilisables, magasin d'affinité.
2. `GET /internal/nodes` (jeton) : raison par nœud (`CIRCUIT_OPEN`, `MODEL_NOT_OBSERVED`, …).
3. Corriger la Gateway (modèle installé, capacité activée) ; le Pool réobserve au prochain cycle de
   sonde (`probe_interval_seconds`) ou au redémarrage.
4. Redémarrage : `docker compose restart pool` — stateless.

## Limites connues

Les états du disjoncteur et les compteurs en vol sont **par processus** : plusieurs réplicas du Pool
ont chacun leur vue. L'affinité ASR est la seule donnée partagée.
