# Proposal

## Why

Une ligne n'a parfois qu'une classe parmi plusieurs, comme les trois
espèces de l'iris. La tâche binaire refuse cette cible : elle attend
une valeur dans `[0, 1]` et une forme identique aux logits. Il faut une
tâche qui compare un vecteur de logits à un indice de classe.

## What Changes

- Tâche `multiclass`. Le modèle continue de renvoyer des logits. La
  tâche applique un softmax, une entropie croisée sur ces logits, et
  décide par l'indice du plus grand logit. Le seuil du module ne change
  pas cette classe.
- Le nombre de classes est le nombre de logits. La cible est une
  colonne : des indices entiers de `0` inclus à ce nombre exclus. Une
  valeur hors de cet intervalle, une cible non entière, ou moins de
  deux logits, est refusée avant la perte.
- Métriques de l'époque : justesse, précision, rappel et F1 macro sur
  toutes les classes du modèle, coefficient de Matthews multiclasse,
  matrice de confusion carrée (effectifs et ligne normalisée). Pas
  d'average precision.
- Fichier de prédiction : une colonne `prediction` (l'indice de classe)
  et une colonne `probability_k` par classe. Les probabilités d'une
  ligne somment à 1.
- Groupe `conf/task/multiclass.yaml` (`name: multiclass`,
  `loss: cross_entropy`). La configuration par défaut reste `binary`.
  `conf/train_mlp.yaml` choisit déjà `task: multiclass` : cette ligne
  et les autres clefs du fichier ne sont pas modifiées.
- La tâche binaire, ses métriques (`ap`, `mcc`) et le CSV à une sortie
  (`id`, `probability`, `prediction`) ne changent pas.

Hors périmètre : multilabel, segmentation, régression, pondération des
classes, lissage d'étiquettes, cible en one-hot, seuil appris.

## Capabilities

### New Capabilities
- `multiclass-task`: mesurer et entraîner une classification où chaque
  ligne a exactement une classe parmi K, K étant le nombre de logits.

### Modified Capabilities

Aucune : `openspec/specs/` est vide. Le comportement binaire déjà
décrit par `introduce-binary-task` n'est pas modifié.

## Impact

- `ia_xp_plateform/task/` : nouvelle tâche, enregistrée dans
  `create_task` à côté de `binary`.
- `ia_xp_plateform/metric/` : compteurs multiclasses et figure de
  matrice K×K. La matrice 2×2 binaire reste.
- `ClassificationModule` journalise les clefs que la tâche fournit.
  Sans tâche nommée, il reste binaire.
- `predict` écrit le CSV multiclasse quand la tâche renvoie un indice
  et plusieurs probabilités. `Predictor` délègue la décision à la tâche.
- `conf/task/multiclass.yaml`. Aucune dépendance nouvelle. Les
  checkpoints déjà écrits se rechargent : le softmax n'est pas un poids.
