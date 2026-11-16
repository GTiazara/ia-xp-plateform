# Proposal

## Why

`test/unitaire/test_configuration.py` compare la configuration Hydra
composée à des nombres écrits en dur (`input_features == 1`,
`max_epochs == 200`, `hidden_layers == 1`, `learning_rate == 0.01`,
`early_stopping_patience == 5`). Ces nombres sont des réglages
d'expérience que l'utilisateur change dans `conf/`. La suite est déjà
rouge pour cette raison : `config.yaml` a `input_features: 16` et
`train_mlp.yaml` a `hidden_layers: 10`.

## What Changes

- Les tests de `test/unitaire/test_configuration.py` qui lisent la
  configuration par défaut, le groupe `model=mlp`, les fichiers nommés
  `train_perceptron` / `train_mlp`, ou le namespace aplati, ne comparent
  plus un champ numérique à un littéral venant d'un YAML d'expérience.
- Ils vérifient à la place que le champ existe, a le bon type, et
  respecte sa contrainte (entier strictement positif, flottant
  strictement positif, ou `None` quand le YAML laisse la commande
  déduire la valeur).
- Les tests qui posent eux-mêmes la surcharge (`seed=7`,
  `trainer.max_epochs=3`, `data.data_path=jeu.csv`) continuent
  d'affirmer la valeur qu'ils ont envoyée, ainsi que l'interpolation
  `${seed}` vers `data.split_seed`.
- Aucun fichier de `conf/` n'est modifié. Le comportement de
  `ia-xp-conf`, des factories et de l'entraînement ne change pas.

## Capabilities

### New Capabilities

Aucune. Le changement ne modifie pas un comportement observable de
l'application.

### Modified Capabilities

Aucune. `openspec/specs/` est vide. Le marqueur `skip_specs: true` est
posé dans `.openspec.yaml` : il s'agit d'un correctif de tests, pas
d'une exigence produit.

## Impact

- `test/unitaire/test_configuration.py` uniquement.
- Les tests d'intégration Hydra (`test/integration/test_hydra_flow.py`)
  imposent déjà leurs propres surcharges (`model.input_features=2`,
  `trainer.max_epochs=1`) : ils ne sont pas dans le périmètre.
- README : une phrase dans la section Tests, pour dire que la
  composition Hydra est vérifiée sans figer les nombres des fichiers
  d'expérience.
