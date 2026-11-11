---
title: POC Moodle — abandon à 28 jours
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.28.0
python_version: "3.13"
app_file: app.py
short_description: Démonstration sur données synthétiques avec explications SHAP
---

# Risque d’abandon à 28 jours — démonstration Moodle

Cette application pédagogique estime le risque d’abandon administratif d’une inscription
dans les 28 jours suivant une photographie fixée au **30 juin 2026**.
Elle utilise uniquement des **données synthétiques** ; les identifiants sont fictifs.

Deux onglets permettent de rechercher un stagiaire et de consulter une vue globale
de la cohorte. Les exemples proposés donnent des identifiants valides.
Les graphiques SHAP sont calculés sur le serveur, sans réentraînement.
Le risque et la référence du graphique individuel sont exprimés en pourcentage ;
les contributions individuelles et globales sont exprimées en points de pourcentage
(+5 points fait passer, par exemple, de 10 % à 15 %).
L’explication utilise TreeSHAP en probabilité, avec une référence reproductible de
100 inscriptions du groupe étudié. Ce calcul diffère du SHAP en log-odds du notebook ;
les prédictions restent inchangées. Si TreeSHAP ne reconstitue pas la prédiction
à la tolérance de 0,000001, les seuls cas concernés sont recalculés par SHAP avec
10 permutations aller-retour et la même référence. Ces contributions sont alors
estimées ; leur somme est également contrôlée avant affichage.
Pour une personne ayant plusieurs inscriptions, le graphique individuel explique
l’inscription au risque maximal.

Les formations proposées sont :

- Concevoir et animer une formation pour adultes ;
- Excel : maîtriser les fonctions essentielles ;
- Réussir sa prise de fonction de manager.

## Portée des résultats

Sur les deux populations synthétiques réunies : 1 877 inscriptions évaluables,
94,8 % de précision, 85,9 % de rappel et 90,1 % de F1.
Les F1 des deux populations sont respectivement 95,0 % et 85,4 %.
Ces résultats dépendent des hypothèses de simulation et ne constituent pas
une validation sur des données réelles. Les risques ne sont pas calibrés sur le terrain.
SHAP décrit les associations apprises, sans établir de causalité.
Aucune alerte automatique n’est déclenchée.

## Exécution

Python 3.13 et les dépendances de `requirements.txt` sont nécessaires.
Depuis la racine du dossier :

```sh
python app.py
```

Le fichier `demo.parquet` contient les données synthétiques, le modèle entraîné,
les métriques et la liste des dépendances dans ses métadonnées.
Le workflow GitHub Actions ajoute automatiquement `requirements.txt` au Space.
Les fichiers `MAPPING.md`, les scripts d’export, les sources Moodle et les notebooks
ne sont pas nécessaires à son exécution.
