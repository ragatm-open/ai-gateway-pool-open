# AI Gateway Pool Open

Routeur HTTP **autonome** placé devant une ou plusieurs instances d'**AI Gateway Open**, sans
modifier le contrat d'inférence consommé par RAGatm Open (`gateway-api/1.0`) : les routes
`/v1/*` et `/health/*` de la Gateway sont relayées octet pour octet, les erreurs de la Gateway
conservent leur signification, et les refus propres au Pool sont marqués (`X-AIGP-Error`,
`{"error": {...}}`). Aucun GPU, aucune inférence dans le Pool.

```text
client (RAGatm Open) ──► Pool Open ──► AI Gateway Open ──► Ollama / TEI
                           └─► Redis (affinité de jobs, multi-nœuds seulement)
```

Le Pool est **facultatif** pour RAGatm Open Compact : l'accès direct à la Gateway reste le
profil nominal ; le Pool ajoute une couche d'orchestration (sélection de nœud, disjoncteur,
allowlist d'upstreams, affinité, saturation) sans changement de contrat applicatif.

Statut : mono-nœud qualifié avec une Gateway Open réelle ; multi-nœuds préparé (nœuds
simulés + Redis réel), **non qualifié sur plusieurs GPU** (`MULTINODE_HARDWARE_QUALIFIED = NO`).

## Démarrage rapide (mono-nœud)

Module **facultatif** : RAGatm Open et le kit Compact fonctionnent en déploiement Direct sans Pool ; le Pool
s'ajoute séparément devant une ou plusieurs instances d'[ai-gateway-open](https://github.com/ragatm-open/ai-gateway-open).

```bash
git clone https://github.com/ragatm-open/ai-gateway-pool-open.git && cd ai-gateway-pool-open
python3.12 -m venv .venv && .venv/bin/pip install --require-hashes -r requirements.lock && .venv/bin/pip install --no-deps -e .
cp config/compact.example.yaml pool.yaml           # relire chaque valeur ; `status: approved` pour un déploiement
export POOL_ENVIRONMENT=production POOL_CONFIG_FILE=$PWD/pool.yaml POOL_ADMIN_TOKEN=$(openssl rand -hex 32)
.venv/bin/ai-gateway-pool                           # écoute 127.0.0.1:8080 ; relaie vers http://ai-gateway:8100
```

Avec Docker Compose : `deploy/compose/pool/` (Pool + Redis, Gateway Open fournie par ailleurs) ;
environnement de qualification Direct/Pool : `deploy/compose/qualification/`.

## Documentation

`docs/` — architecture, contrat Gateway (`gateway-contracts.md`), relais HTTP, flux NDJSON,
politiques de routage, registre des nœuds, multi-nœuds (exemple), affinité ASR, basculement,
sécurité, performance, reprise, qualification du contrat, CI locale.

## Vérifications locales

```bash
scripts/check-local.sh      # lint, types, validation des configs, secrets, tests (Redis jetable), démarrage
scripts/validate-config.sh  # validation hors ligne des configurations versionnées
```

## Licence

Code original sous [Apache License 2.0](LICENSE) (voir [`NOTICE`](NOTICE)) ; composants tiers : [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Le Pool est **facultatif** : RAGatm Open ([ragatm-open](https://github.com/ragatm-open/ragatm-open)) fonctionne en accès direct à [ai-gateway-open](https://github.com/ragatm-open/ai-gateway-open). Contribution : [`CONTRIBUTING.md`](CONTRIBUTING.md) ; sécurité : [`SECURITY.md`](SECURITY.md).
