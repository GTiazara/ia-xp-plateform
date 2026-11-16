# Design

## Context

Voir `proposal.md` (Why) et la spec `binary-classification-metrics`.
État observé :

- `BinaryClassificationMetrics` compte, en flottants Python, les vrais
  positifs, faux positifs, faux négatifs, les bonnes prédictions et le
  total. `compute()` rend `acc`, `precision`, `recall`, `f1`. Ces
  flottants ne sont pas des buffers : ils ne partent pas dans le
  checkpoint.
- `ClassificationModule._log_metrics` journalise chaque clef de
  `compute()` sous `{stage}_{clef}`. Ajouter une clef suffit à la
  publier. Le seuil vient du module (`0.5` par défaut) et est passé à
  `update`.
- `update` compare déjà élément par élément : deux sorties d'une même
  ligne sont deux décisions binaires.
- Aucune dépendance de métriques au-delà de `torch`. `openspec/specs/`
  est vide.

## Goals / Non-Goals

**Goals:**
- L'AP et le MCC sortent de `compute()` et apparaissent dans TensorBoard
  sans deuxième chemin de journalisation.
- L'AP reste définie quand aucune probabilité ne franchit le seuil.
- Le MCC vaut 0 pour un modèle qui répond toujours 0.

**Non-Goals:**
- Changer la perte, le seuil, le checkpoint, ou les clefs déjà
  journalisées.
- Courbe précision-rappel, AUC ROC, score de Brier.

## Decisions

### L'AP garde les scores, le MCC non

`update` conserve, en plus des compteurs, les probabilités et les
cibles du lot (`detach`, CPU, une dimension). `compute` les
concatène. `reset` vide les deux listes. Ce ne sont pas des buffers :
le checkpoint ne grossit pas, comme pour les compteurs actuels.

Le MCC n'a besoin que des quatre comptes déjà tenus. Le vrai négatif
est `total - VP - FP - FN`, la même soustraction que
`confusion_matrix()`.

Alternative écartée : ne garder que des histogrammes de scores. Moins
de mémoire, mais l'AP n'est alors qu'approchée, et la spec demande
`5/6` et `1/2` exacts.

Alternative écartée : `torchmetrics` ou `sklearn`. Dépendance nouvelle
pour deux formules courtes, et un objet d'état de plus à côté des
compteurs maison.

### Average precision non interpolée, tri stable

On trie par probabilité décroissante avec un tri stable
(`torch.argsort(..., stable=True)`). À chaque cible positive, la
précision est `positifs vus / rang`. L'AP est la moyenne de ces
précisions. Pas d'interpolation à 11 points : elle ne donne pas `5/6`
sur l'exemple de la spec.

Deux scores égaux gardent l'ordre d'arrivée, parce que les lots sont
concaténés dans l'ordre des `update`. `torch.argsort` sans `stable`
n'a pas cet ordre : le scénario « scores égaux → AP = 1/2 » deviendrait
aléatoire.

Aucun positif, ou aucun exemple : AP = 0, jamais `NaN`.

### Le MCC suit le seuil déjà passé à `update`

Les comptes VP, FP, FN viennent du seuil reçu (celui du module). Le
MCC ne choisit pas un autre seuil. Formule de la spec ; dénominateur
nul → 0. Cela couvre « toujours 0 » (`VP + FP = 0`) et « toujours 1 »
(`VN + FN = 0`).

### Une sortie, un exemple

`reshape(-1)` sur les probabilités et les cibles, comme le font déjà
les comparaisons élément par élément. Une ligne à deux sorties compte
pour deux exemples binaires. On ne moyenne pas une AP par sortie : le
module n'a qu'une clef `ap`.

### Journalisation inchangée dans sa forme

`compute()` ajoute `ap` et `mcc`. `_log_metrics` itère déjà les clefs,
donc `val_ap` et `val_mcc` apparaissent sans nouvel appel à `self.log`.
Les docstrings du module et de la classe listent ces noms. Le README
explique les deux mesures dans la section du module de classification.

## Risks / Trade-offs

- [Une époque d'entraînement conserve toutes ses probabilités] → sur le jeu actuel, environ 30 000 scalaires. Négligeable. On ne les garde pas d'une époque à l'autre : `reset` les libère.
- [L'AP à 1 et le F1 à 0 en même temps surprennent] → c'est le cas de la spec « classement parfait sous le seuil ». Le README le dit explicitement.
- [Deux sorties indépendantes partagent une seule AP] → cohérent avec la précision actuelle, qui mélange aussi les sorties. Une AP par sortie attendrait un autre contrat de métriques.
- [Le tri stable dépend de l'ordre des lots] → l'ordre du `DataLoader` de validation est fixe (`shuffle=False`). L'entraînement est mélangé : l'AP y dépend de l'ordre seulement en cas d'égalité parfaite de scores.

## Migration Plan

Aucune migration. Les nouvelles courbes apparaissent à côté des
anciennes. Un checkpoint déjà écrit se relit tel quel. Retour arrière :
retirer les deux clefs de `compute()`.
