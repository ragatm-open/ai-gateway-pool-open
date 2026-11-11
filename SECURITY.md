# Politique de sécurité

## Signaler une vulnérabilité

Merci de **ne pas** ouvrir d'issue publique pour une vulnérabilité. Écrivez à **contact@ragatm.com** (canal privé de signalement)
en indiquant : le dépôt et la version concernés (tag ou empreinte de l'archive), une description, les étapes de
reproduction, l'impact estimé et, si possible, une proposition de correctif. Chiffrez le message si vous le jugez
nécessaire et proposez une clé publique dans votre premier envoi.

## Ce à quoi vous pouvez vous attendre

- Accusé de réception et premier avis sur la recevabilité du signalement, dans la mesure des moyens d'un projet
  maintenu bénévolement : le projet **ne s'engage pas sur un délai contractuel** de correction.
- Travail du correctif en privé, publication d'une version corrigée accompagnée d'une note dans `CHANGELOG.md`,
  et mention de la personne ayant signalé (sauf demande contraire).
- Divulgation coordonnée : nous demandons de tenir le détail confidentiel jusqu'à la publication du correctif, et au
  plus tard 90 jours après le signalement, sauf accord mutuel sur un autre calendrier.

## Versions supportées

Seule la dernière version publiée de la branche principale (`main`) reçoit des correctifs de sécurité. Les archives
candidates (`rcN`) et les versions antérieures ne sont pas maintenues.

## Périmètre

Dans le périmètre : le code source de ce dépôt, ses Dockerfiles et fichiers Compose, ses dépendances directes telles
qu'épinglées par les fichiers de verrouillage. Hors périmètre : les images de base et composants tiers (signaler à
leurs mainteneurs), les modèles d'IA et leurs comportements, les déploiements et infrastructures de tiers, ainsi que
les problèmes nécessitant un accès déjà privilégié à la machine hôte.

## Mises à jour de dépendances

Les dépendances sont épinglées (empreintes) et les images de base référencées par digest : toute mise à jour est un
changement explicite, relu et testé (voir `CONTRIBUTING.md`). Les rapports de scan de vulnérabilités des images sont
publiés avec chaque release lorsqu'ils existent ; une vulnérabilité sans correctif amont est documentée, jamais masquée.
