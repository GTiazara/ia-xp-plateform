# Proposal

## Why

La classification est binaire et câblée : `Perceptron` et `MLP` appliquent
une sigmoïde, et `ClassificationModule` appelle toujours `BCELoss`. Une
cible hors de `{0, 1}`, ou une sortie dont la forme ne colle pas à la
cible, part dans le noyau CUDA et s'arrête sur un assert. Avant d'ajouter
une tâche multiclasse, il faut que la perte, l'activation et la décision
appartiennent à une tâche, et que le modèle ne renvoie plus qu'un logit.

## What Changes

- **BREAKING** : `Perceptron.forward` et `MLP.forward` renvoient le logit,
  sans sigmoïde. Le nombre de paramètres et les poids d'un checkpoint
  déjà écrit ne changent pas.
- Une tâche `binary` possède la sigmoïde, la perte `BCEWithLogitsLoss`
  et la décision au seuil déjà configuré. `ClassificationModule.forward`
  et `Predictor.predict_proba` continuent de renvoyer une probabilité
  entre 0 et 1. Les colonnes du CSV de `predict` ne changent pas.
- Avant la perte, la tâche refuse une cible hors de `[0, 1]` ou une
  forme différente de celle des logits, par une `ValueError` qui nomme
  le problème.
- Groupe Hydra `task`, fichier `binary` (`name: binary`, `loss: bce`),
  choisi par défaut. `learning_rate` et `threshold` restent dans
  `module`. La commande argparse `ia-xp` n'a pas de nouveau drapeau :
  elle construit la tâche binaire.
- README : le modèle renvoie un logit ; la tâche en fait une
  probabilité, une perte et une classe.

Hors périmètre : tâche multiclasse, multilabel, segmentation,
régression, `pos_weight`, perte focale, seuil appris, déplacement de
`threshold` hors de `module`.

## Capabilities

### New Capabilities
- `model-logits`: `Perceptron` et `MLP` renvoient un logit de la forme
  `(batch, output_features)`, avec les mêmes paramètres qu'aujourd'hui.
- `binary-task`: la tâche binaire transforme ces logits en probabilités,
  en perte stable et en classe, et refuse une cible incompatible avant
  d'appeler la perte.

### Modified Capabilities

Aucune : `openspec/specs/` est vide. Les métriques binaires déjà
décrites par la change `add-average-precision-and-mcc` reçoivent
toujours des probabilités ; leur définition ne change pas.

## Impact

- `ia_xp_plateform/models/perceptron.py`, `models/mlp.py` et leurs
  tests : la sortie n'est plus bornée dans `[0, 1]`.
- Nouveau paquet `ia_xp_plateform/task/` et factory `create_task`.
- `ClassificationModule` et `Predictor` délèguent perte, sigmoïde et
  décision à la tâche. Les tests qui comparent `model(x)` à une
  probabilité comparent désormais `sigmoid(model(x))`.
- `conf/task/binary.yaml`, défaut ajouté dans `config.yaml`,
  `train_perceptron.yaml` et `train_mlp.yaml` sans modifier les autres
  clefs de ces fichiers. `namespace_from_config` expose le nom et la
  perte.
- README, sections des modèles, du module et du prédicteur.
- Aucune dépendance nouvelle. Un checkpoint déjà écrit se relit : la
  sigmoïde n'était pas un paramètre.
