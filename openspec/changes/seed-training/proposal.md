# Proposal

## Why

Deux lancements de la même expérience ne suivent pas le même chemin.
La graine ne mélange que les lignes du découpage. Les poids naissent
du générateur global, et l'ordre des batches aussi. `val_loss` peut
donc choisir un autre checkpoint sans qu'aucun réglage n'ait changé.

## What Changes

- `run_train` sème l'entraînement avec la graine de l'expérience avant
  de construire le modèle. Même graine, mêmes données, même
  entraînement sur CPU : mêmes poids à la fin.
- Le découpage train / val / test garde son générateur privé. La même
  graine continue de mettre les mêmes lignes dans les mêmes parts.
- Le README et l'aide de `--seed` disent que cette graine sème aussi
  l'initialisation et l'ordre des batches. Le commentaire de `seed`
  dans `conf/config.yaml` suit. Aucune valeur de `conf/` ne change.
- `test` et `predict` ne sèment pas l'entraînement : ils ne créent pas
  de poids. Les tests qui construisent un `Trainer` eux-mêmes restent
  non semés.

Hors périmètre : algorithmes déterministes de PyTorch, courbes bit à
bit identiques sur GPU, graine des workers déjà supérieure à zéro
dans une expérience, multilabel, segmentation.

## Capabilities

### New Capabilities

- `training-seed`: la commande d'entraînement reproduit les poids sur
  CPU à partir de la graine de l'expérience, sans changer le découpage.

### Modified Capabilities

Aucune : `openspec/specs/` est vide.

## Impact

- `ia_xp_plateform/cli/command/train.py` : un appel à
  `pytorch_lightning.seed_everything` en tête de `run_train`. Les deux
  entrées (`ia-xp` et `ia-xp-conf`) passent déjà par cette fonction.
  `args.seed` vient de `--seed` ou de `split_seed`, lui-même `${seed}`.
- README, aide argparse, commentaire de `conf/config.yaml`. Pas de
  nouvelle dépendance. Le fichier de split et les checkpoints déjà
  écrits ne changent pas de format.
