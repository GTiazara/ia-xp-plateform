# Design

## Context

Voir `proposal.md` (Why) et la spec `multiclass-task`. État observé :

- `Perceptron` et `MLP` renvoient des logits. `BinaryTask` applique la
  sigmoïde, `BCEWithLogitsLoss`, et un seuil. `create_task` ne connaît
  que `"binary"`, et son argument `loss` vaut `"bce"` par défaut.
- `ClassificationModule` construit toujours
  `BinaryClassificationMetrics`. `_step` appelle
  `metrics.update(probabilities, y, threshold)`. `forward` publie
  `task.probabilities`.
- La cible tabulaire est un flottant de forme `(N, nombre de colonnes)`.
  Une colonne donne `(N, 1)`.
- `confusion_matrix_figure` dessine une matrice 2×2 : les graduations
  sont fixées à `0` et `1`. `normalize_confusion_matrix` divise déjà
  chaque ligne par son total, quelle que soit la taille.
- `run_predict` ne passe pas la tâche au `Predictor`. Il recalcule la
  classe par `probabilities >= threshold`, puis écrit une colonne par
  sortie quand il y en a plusieurs.
- `conf/train_mlp.yaml` contient déjà `task: multiclass` et
  `output_features: 3`. La configuration par défaut reste `task: binary`.

## Goals / Non-Goals

**Goals:**
- La tâche multiclasse est le seul endroit qui connaît le softmax,
  l'entropie croisée et l'argmax.
- Le module journalise les clefs que la tâche lui donne, sans chemin
  spécial pour `ap`.
- Une cible incompatible échoue en Python avant le critère.

**Non-Goals:**
- Accepter une cible one-hot.
- Changer la perte, les métriques ou le CSV de la tâche binaire.
- Modifier les clefs déjà présentes de `train_mlp.yaml`.

## Decisions

### `MulticlassTask` à côté de `BinaryTask`

`ia_xp_plateform/task/multiclass.py` expose `MulticlassTask`. Elle n'est
pas un `nn.Module`.

- `probabilities(logits)` renvoie `softmax` sur la dernière dimension.
- `predict(logits, threshold)` renvoie `argmax` sur cette dimension,
  avec `keepdim=True`, donc une forme `(N, 1)`. `threshold` est accepté
  et ignoré, pour garder la même signature que la tâche binaire.
- `loss(logits, targets)` vérifie le lot, puis appelle
  `nn.CrossEntropyLoss` (réduction moyenne, sans poids). Le critère
  n'a pas de paramètre : un checkpoint ne gagne aucune clef.

`TASKS` gagne `"multiclass"`. `create_task(name, loss=None)` appelle
`task_cls()` quand `loss` est absent, sinon `task_cls(loss=loss)`.
`BinaryTask()` reste `bce`. `MulticlassTask()` est `cross_entropy`.
Ainsi `create_task("multiclass")` ne reçoit pas `"bce"` par accident.
Une perte refusée cite `cross_entropy`. Un nom absent cite `binary` et
`multiclass`.

Alternative écartée : réutiliser `BCEWithLogitsLoss` sur un one-hot.
La cible du fichier est un indice, et l'entropie croisée sur les logits
est la perte stable de ce cas.

### La vérification précède le critère

`loss` exige des logits de forme `(N, K)` avec `K >= 2`. Sinon la
`ValueError` nomme le nombre de logits.

La cible `(N, 1)` est ramenée à `(N,)`. `(N,)` est acceptée. Toute
autre forme est refusée en nommant les deux formes.

Chaque valeur doit être à moins de `1e-5` de son entier le plus
proche, puis cet entier doit être dans `0 .. K-1`. `2.0` devient
l'indice 2. `1.5` et `3` (quand `K` vaut 3) sont refusés avant
`CrossEntropyLoss`. L'intervalle nommé va de 0 à `K - 1`. La tolérance
couvre un indice stocké en `float32` ; elle ne laisse pas passer un
demi-entier.

`CrossEntropyLoss` reçoit les logits et ces indices en `long`.

### Les métriques appartiennent à la tâche

`MulticlassClassificationMetrics` compte, sur l'époque, une matrice
`K×K` en flottants Python. `K` est le nombre de colonnes de la première
probabilité. `reset` vide cette matrice. Ce ne sont pas des buffers.

`update` prend des probabilités `(N, K)` et des cibles déjà réduites à
des indices `(N,)`. La classe prédite est `argmax`. `compute` rend
`acc`, `precision`, `recall`, `f1` et `mcc`. Pas de clef `ap`.

La précision, le rappel et le F1 d'une classe suivent les mêmes
quotients que le binaire, avec 0 si le dénominateur est nul. La valeur
publiée est la moyenne des `K` classes, y compris une classe jamais vue.
Le MCC est la formule de la spec. Un dénominateur nul donne 0.

Chaque tâche gagne `new_metrics()` et
`observe(metrics, probabilities, targets, threshold)`. La tâche binaire
délègue à `BinaryClassificationMetrics.update` avec le seuil. La tâche
multiclasse ignore le seuil, réduit la cible, puis appelle `update`.
Le module construit `self.metrics` à partir de `task.new_metrics()` et
n'appelle plus `update` lui-même. `_log_metrics` itère toujours les
clefs de `compute()`, donc `val_ap` n'apparaît que pour la tâche
binaire.

`confusion_matrix_figure` lit la taille de la matrice pour les
graduations `0 .. K-1`. Le dessin 2×2 binaire n'est plus un cas à part.
La normalisation de ligne reste la fonction déjà écrite.

Alternative écartée : un F1 micro, qui se confond avec la justesse
quand chaque ligne a une classe. Le macro sépare une classe absente,
ce que la spec demande avec `4/9`.

### Le CSV suit la forme renvoyée par la tâche

`run_predict` construit la tâche comme `run_train` (`build_task`) et la
passe à `Predictor.from_checkpoint`. Les classes viennent de
`predictor.predict`, plus de la comparaison au seuil.

- une probabilité et une classe : colonnes `id`, `probability`,
  `prediction` ;
- plusieurs probabilités et une seule classe `(N, 1)` : `id`,
  `prediction`, `probability_0` … `probability_{K-1}` ;
- autant de classes que de probabilités : le CSV binaire à plusieurs
  sorties, inchangé.

L'argparse n'a pas de drapeau de tâche. Sans attribut `task`,
`build_task` rend `None` et le prédicteur reste binaire.

### La configuration ajoute un fichier, pas un défaut

`conf/task/multiclass.yaml` contient `name: multiclass` et
`loss: cross_entropy`. Aucune ligne de `config.yaml`,
`train_perceptron.yaml` ou `train_mlp.yaml` n'est éditée.
`train_mlp.yaml` pointe déjà vers ce fichier.

Le test de la configuration par défaut continue de voir `binary`. Un
test charge la surcharge `task=multiclass` et vérifie le nom et la
perte. Les configurations nommées ne figent pas ce nom.

Le test d'intégration qui entraîne `train_mlp` sur le jeu binaire à une
colonne `0`/`1` surcharge `task=binary`. Ce jeu n'est pas multiclasse ;
la surcharge appartient au test, pas au fichier d'expérience.

## Risks / Trade-offs

- [Une classe absente vaut 0 dans le F1 macro] → c'est le scénario `4/9`. Une époque courte semble plus mauvaise qu'un F1 micro. Le README le dit.
- [L'égalité de logits choisit le plus petit indice] → `argmax` est déterministe. Pas de tirage.
- [Un indice `float32` très éloigné de son entier est refusé] → la tolérance `1e-5` accepte `2.0` et refuse `1.5`.
- [Prédire un checkpoint multiclasse sans la tâche softmaxe mal] → `run_predict` prend la tâche de la configuration. Le README dit de garder la même tâche qu'à l'entraînement.
- [`train_mlp` sur l'iris démarre vraiment une fois le fichier présent] → les autres clefs du fichier restent celles de l'expérience. Cette change ne lance pas cet entraînement.

## Migration Plan

Aucune conversion de checkpoint. Le softmax n'est pas un poids.
Retour arrière : retirer `multiclass` de `TASKS` et le fichier
`conf/task/multiclass.yaml`. `train_mlp.yaml` redevient alors une
configuration que Hydra ne sait plus assembler, parce qu'il nomme déjà
cette tâche.
