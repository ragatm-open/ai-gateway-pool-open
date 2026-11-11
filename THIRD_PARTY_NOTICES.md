# Mentions de tiers — AI Gateway Pool Open

Ce fichier recense les composants tiers utilisés, référencés ou redistribués par cette distribution, avec les licences publiées par leurs éditeurs (métadonnées de paquets, fiches de modèles). Il ne constitue pas une analyse juridique. Le code original de ce dépôt est sous **Apache-2.0** (`LICENSE`, `NOTICE`) ; les composants ci-dessous conservent leurs licences respectives.

## Composants soumis à une obligation particulière

| Composant | Licence | Disposition |
|---|---|---|
| certifi | MPL-2.0 | fichier de certificats non modifié (dépendance transitive de httpx) |
| pathspec | MPL-2.0 | outillage de développement uniquement |
| `python:3.12-slim` (image de base) | PSF-2.0 + licences Debian | référencée par digest, non modifiée |
| `redis:7.2-alpine` (qualification multi-nœuds) | BSD-3-Clause | référencée par digest ; non redistribuée |

## Bibliothèques Python (`requirements.lock` = runtime ; `requirements-dev.lock` = développement)

| Paquet | Version | Licence (métadonnées publiées) | Portée |
|---|---|---|---|
| annotated-doc | 0.0.5 | MIT | runtime |
| annotated-types | 0.8.0 | MIT | runtime |
| anyio | 4.16.0 | MIT | runtime |
| ast_serialize | 0.13.1 | MIT | développement / CI |
| build | 1.6.1 | MIT | développement / CI |
| certifi | 2026.10.23 | MPL-2.0 | runtime |
| click | 8.5.1 | BSD-3-Clause | runtime |
| fastapi | 0.162.1 | MIT | runtime |
| h11 | 0.16.0 | MIT | runtime |
| httpcore | 1.0.9 | BSD-3-Clause | runtime |
| httpx | 0.28.1 | BSD-3-Clause | runtime |
| idna | 3.20 | BSD-3-Clause | runtime |
| iniconfig | 2.3.1 | MIT | développement / CI |
| librt | 0.16.0 | MIT | développement / CI |
| mypy | 2.4.1 | MIT | développement / CI |
| mypy_extensions | 1.2.0 | MIT | développement / CI |
| opentelemetry-api | 1.45.2 | Apache-2.0 | runtime |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | développement / CI |
| pathspec | 1.1.1 | Mozilla Public License 2.0 (MPL 2.0) | développement / CI |
| pluggy | 1.7.0 | MIT | développement / CI |
| pydantic | 2.14.3 | MIT | runtime |
| pydantic-settings | 2.16.2 | MIT | runtime |
| pydantic_core | 2.50.4 | MIT | runtime |
| Pygments | 2.21.0 | BSD-2-Clause | développement / CI |
| pyproject_hooks | 1.3.3 | MIT | développement / CI |
| pytest | 9.2.0 | MIT | développement / CI |
| pytest-asyncio | 1.4.1.post1 | Apache-2.0 | développement / CI |
| python-dotenv | 1.2.4 | BSD-3-Clause | runtime |
| PyYAML | 6.0.3 | MIT | runtime |
| redis | 8.1.0 | MIT | runtime |
| ruff | 0.17.4 | MIT | développement / CI |
| sniffio | 1.3.1 | MIT OR Apache-2.0 | runtime |
| starlette | 1.10.0 | BSD-3-Clause | runtime |
| types-PyYAML | 6.0.12.20261103 | Apache-2.0 | développement / CI |
| typing-inspection | 0.4.5 | MIT | runtime |
| typing_extensions | 4.16.0 | PSF-2.0 | runtime |
| uvicorn | 0.54.0 | BSD-3-Clause | runtime |
