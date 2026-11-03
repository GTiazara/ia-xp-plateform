# XP-01 — Un perceptron pour décider d'accélérer ou non

## Objectif

Vérifier qu'un perceptron — le plus simple des réseaux de neurones
artificiels — peut apprendre une règle de décision de conduite très simple :
**faut-il accélérer, oui ou non ?**

Un perceptron prend plusieurs entrées, leur attribue des poids, en fait la
somme (+ un biais), puis applique une fonction d'activation pour produire une
décision.

## 1. Entrées / sortie

| Entrée                     | Exemple de valeur |
| --------------------------- | ------------------ |
| Distance à l'obstacle       | 2 mètres           |
| Présence d'un obstacle      | Oui / Non          |
| Position de la voiture      | Gauche / Droite    |
| Vitesse                     | 30 km/h            |

Sortie du perceptron :

- `0` : ne pas accélérer
- `1` : accélérer

Dans le code (`xp.train.data.datasets.car_driving`), ces 4 entrées sont
normalisées dans `[0, 1]` et rangées dans cet ordre (voir `FEATURES`) :

```
0. distance_obstacle_m   (distance / distance_max)
1. obstacle_present       (0 ou 1)
2. position               (0 = gauche, 1 = droite)
3. speed_kmh              (vitesse / vitesse_max)
```

## 2. Comment lui apprendre à conduire ?

Trois étapes, dans l'ordre où elles sont implémentées ici :

1. **Collecter des données** : pour cette XP, on simule différentes
   situations de conduite avec un jeu de données synthétique
   (`CarDrivingDataset`) plutôt que de les enregistrer sur une vraie voiture.
2. **Entraînement** : on fournit au perceptron les 4 entrées et la décision
   attendue (calculée par une règle "experte" simulée — voir
   `_expert_rule` dans `car_driving.py`), avec un peu de bruit pour simuler
   des situations ambiguës.
3. **Apprentissage** : l'optimiseur (`SGD`) ajuste les poids du perceptron
   pour réduire l'erreur entre sa décision et la décision attendue
   (`BCEWithLogitsLoss`).

> Pour cette première itération de la plateforme, on se concentre sur
> l'**architecture** (modèle / tâche d'apprentissage / données / config /
> CLI) plutôt que sur le point 4 ci-dessous (la petite voiture virtuelle qui
> évite des murs) : ce sera l'objet d'une prochaine XP.

## 3. Est-ce suffisant pour conduire une vraie voiture ?

Non. Un perceptron simple possède des limites importantes :

- Il ne peut apprendre que des relations **relativement simples** (il ne
  sépare bien que des données à peu près **linéairement séparables** — c'est
  d'ailleurs pour cela que la règle experte simulée ici est volontairement
  simple).
- Il ne comprend pas réellement des images complexes.
- Il ne gère pas correctement toutes les situations imprévues.

Pour un système de conduite autonome réel, on utilise plutôt :

- des **réseaux de neurones multicouches (MLP)** ;
- des **réseaux convolutifs (CNN)** pour analyser des images ;
- de l'**apprentissage par renforcement** pour apprendre des stratégies de
  conduite complètes.

## 4. Idée pour une prochaine XP (XP-02 ?)

Une petite voiture virtuelle dans laquelle :

- les entrées sont la distance aux murs et la position de la voiture ;
- le réseau de neurones (MLP, pas un simple perceptron) décide de tourner à
  gauche, à droite, ou d'avancer ;
- le programme apprend progressivement à éviter les obstacles.

Cette idée est volontairement **documentée mais pas implémentée** dans cette
XP : elle correspond à l'étape suivante une fois l'architecture validée sur
le cas le plus simple (un perceptron, une décision binaire).

## En résumé

Un perceptron est un excellent point de départ pour comprendre comment une
intelligence artificielle apprend à prendre des décisions, mais il est trop
limité pour piloter seul une vraie voiture.

## Comment lancer l'XP

```bash
# Entraînement (config par défaut : conf/train_xp01_perceptron.yaml)
poetry install
poetry run python xp/main.py

# Avec quelques overrides Hydra
poetry run python xp/main.py trainer.max_epochs=20 datamodule.n_samples=5000
```

Les métriques (`loss`, `accuracy`, `precision`, `recall`, `f1`) sont
loggées sur TensorBoard dans `${LOG_DIR}` (voir `.env.dist`).
