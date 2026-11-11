# Contribuer à AI Gateway Pool Open

Merci de votre intérêt. Ce projet est publié sous **Apache License 2.0** : en soumettant une contribution, vous
acceptez qu'elle soit distribuée sous cette licence (article 5 de la licence). Indiquez vos modifications de façon
lisible dans la description de la pull request.

## Règles

1. **Aucune donnée réelle** dans les tests, fixtures, exemples ou documentation : organisations fictives, documents
   synthétiques, adresses d'exemple (`example.org`), jamais de secret ni de donnée personnelle.
2. **Aucun fournisseur d'IA externe** : l'édition Open fonctionne exclusivement avec des modèles exécutés localement.
3. **Dépendances épinglées** : toute évolution passe par les fichiers de verrouillage (empreintes) et les images de base
   par digest ; décrivez la raison de la mise à jour.
4. **Tests** : une modification de comportement est accompagnée de tests ; un test qui échoue n'est jamais supprimé
   pour « verdir » la suite.
5. **Style** : Python formaté/vérifié par `ruff`, typage vérifié quand le module le prévoit ; frontend : `tsc --noEmit`.
6. **Sécurité** : voir `SECURITY.md` ; ne signalez pas de vulnérabilité par une issue publique.

## Exécuter les tests

```bash
python3.12 -m venv .venv && .venv/bin/pip install --require-hashes -r requirements-dev.lock && .venv/bin/pip install --no-deps -e .
.venv/bin/python -m pytest -q tests/unit tests/contracts   # profil FAST : unités + contrats (≈ 12 s, aucun service)
.venv/bin/python -m pytest -q -m "not redis_real"           # profil STANDARD : suite complète sans Redis réel (serveurs locaux simulés)
scripts/check-local.sh                                      # profil FULL : lint, typage, suite complète avec Redis réel jetable, démarrage local
```

## Processus

Ouvrez une issue pour discuter d'un changement significatif avant de l'implémenter. Les pull requests sont relues ;
les profils de tests `fast` doivent passer sur chaque PR, `standard`/`full` avant une release.
