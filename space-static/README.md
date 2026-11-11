---
title: POC Moodle — abandon à 28 jours
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
short_description: "Abandon à 28 jours : données synthétiques et SHAP"
license: mit
---

# Risque d’abandon à 28 jours — démonstration statique

Recherche par identifiant et explications sur une cohorte entièrement synthétique,
avec une photographie fixée au 30 juin 2026. Les identifiants sont fictifs.

Les risques et les contributions SHAP sont calculés et vérifiés lors de l’export.
Le navigateur affiche les résultats de cette photographie ; il ne lance pas le modèle.
Une modification du jeu de données nécessite un nouvel export et une publication.
Pour un stagiaire ayant plusieurs inscriptions, l’explication porte sur son inscription
au risque maximal. Toutes les inscriptions du stagiaire restent consultables.

SHAP est exprimé en points de pourcentage et explique des associations apprises,
sans établir de causalité. Le calcul utilise TreeSHAP en probabilité avec une référence
de 100 inscriptions. Les cas ne reconstituant pas la prédiction sont recalculés
par permutations, avec contrôle de l’additivité avant export.
Les risques ne sont pas calibrés sur le terrain et aucune alerte automatique n’est déclenchée.

## Publication

Deux fichiers suffisent au Space : `index.html` et `README.md`.
Le HTML inclut les données de démonstration et le graphique global ; il n’utilise
aucune dépendance externe, aucun serveur Python et aucune ressource GPU.
Le workflow `.github/workflows/publish-huggingface.yml` reste dans GitHub.
Il utilise le secret `MOODLE_ABANDON` et la variable `HF_SPACE_ID`.

Les sources, le Parquet et le modèle ne sont pas nécessaires sur le Space statique.
L’exporteur reste dans le projet de travail afin de régénérer la démonstration.
