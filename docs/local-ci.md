# CI locale

## Objectif

Reproduire, sans service externe, la porte de qualité que le workflow `.github/workflows/ci.yml`
exécute : lint (`ruff check`, `ruff format --check`), types (`mypy`), validation des configurations,
hygiène des secrets, tests (dont tests Redis réels sur un Redis jetable), cohérence du lock, démarrage
local.

## Principe

`scripts/check-local.sh` crée si besoin un Redis jetable (`redis:7.2-alpine`, mot de passe aléatoire,
port éphémère en boucle locale, supprimé à la sortie), exporte `AIGP_TEST_REDIS_URL` et
`AIGP_TEST_REDIS_CONTAINER` (tests d'indisponibilité par mise en pause du conteneur), puis enchaîne
les étapes. Aucune Gateway réelle n'est contactée : la qualification live est une campagne séparée
(`deploy/compose/qualification/`, `tests/parity/`).

## Mise en œuvre

```bash
python3.12 -m venv .venv
.venv/bin/pip install --require-hashes -r requirements-dev.lock
.venv/bin/pip install --no-deps -e .
scripts/check-local.sh
```

## Limites assumées

Le workflow GitHub Actions fourni est une référence ; l'organisation qui publiera l'édition Open
décidera de l'hébergement de sa CI. Les tests GPU multi-nœuds n'existent pas.
