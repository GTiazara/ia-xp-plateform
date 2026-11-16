# Proposal

## Why

Les modèles reçoivent aujourd'hui les colonnes telles qu'elles sont
lues dans le fichier. Sur `bank-full-ordinal.feather`, `balance` va de
−8019 à 102127 et `duration` de 0 à 4918, à côté d'ordinaux de quelques
unités : le modèle apprend mal et prédit toujours la classe majoritaire
(risque R2 de `doc/audit_phase_a.md`). Il n'existe aucun endroit pour
prétraiter les entrées, ni pour dire que l'entraînement, la validation,
le test et la prédiction ne doivent pas subir les mêmes transformations.

Les données raster viendront après les données tabulaires. Le
prétraitement doit donc être une brique indépendante du format des
données, pour qu'un futur dataset d'images la réutilise sans toucher au
reste.

## What Changes

- Nouvelle brique de transformations d'entrée : un contrat commun
  (`fit`, application sur un exemple ou un lot, état sauvegardable) et
  quatre transformations tabulaires de départ :
  - `standardize` (centrer-réduire par colonne),
  - `min_max` (ramener chaque colonne dans `[0, 1]`),
  - `clip` (borner les valeurs, sans état appris),
  - `gaussian_noise` (bruit aléatoire, pensé pour l'entraînement seul).
- Une factory `create_transform` et un assembleur de pipelines, dans le
  style des autres factories (nom → classe, erreur claire si le nom est
  inconnu).
- Un pipeline par split : `train`, `val`, `test`, `predict`. Les
  transformations qui apprennent des statistiques (`standardize`,
  `min_max`) les apprennent sur le jeu d'entraînement uniquement, puis
  les autres splits réutilisent ces statistiques.
- Les statistiques apprises sont écrites à côté du checkpoint à la fin
  de `train`, relues par `test` et `predict`.
- Nouveau groupe Hydra `conf/augmentation/` : un fichier décrit les
  quatre listes (`train`, `val`, `test`, `predict`). Deux fichiers
  livrés : `none` (aucune transformation, valeur par défaut) et
  `standardize`.
- `TableDataset` et `TableDataModule` acceptent des pipelines ; sans
  pipeline, leur comportement reste identique.
- README : section sur le prétraitement et le groupe `augmentation`.

Hors périmètre : transformations raster, option `--augmentation` pour
le CLI `ia-xp` (argparse), changement du groupe par défaut de
`conf/config.yaml`, mise à l'échelle de la cible `y`.

## Capabilities

### New Capabilities
- `input-transforms`: contrat des transformations d'entrée, leur
  apprentissage sur le jeu d'entraînement, leur état sauvegardable, les
  quatre transformations tabulaires et leur construction par nom.
- `transform-configuration`: groupe `conf/augmentation/`, pipelines par
  split, branchement sur `train`, `test` et `predict`.

### Modified Capabilities

Aucune : `openspec/specs/` est vide.

## Impact

- Code nouveau : `ia_xp_plateform/transform/` et
  `ia_xp_plateform/factory/transform_factory.py`.
- Code modifié : `data/dataset/base.py`, `data/dataset/table_dataset.py`,
  `data/datamodule/table_datamodule.py`, `cli/command/common.py`,
  `cli/command/train.py`, `cli/command/test.py`,
  `cli/command/predict.py`, `cli/from_config.py`.
- Configuration : `conf/augmentation/none.yaml`,
  `conf/augmentation/standardize.yaml`, groupe ajouté aux `defaults` de
  `conf/config.yaml`, `conf/train_mlp.yaml`, `conf/train_perceptron.yaml`.
- Tests : un test unitaire par transformation, par factory et par
  pipeline ; un test d'intégration `fit` → `test` → `predict` avec
  normalisation.
- Aucune dépendance nouvelle (`torch` suffit).
- Les checkpoints existants restent lisibles : sans fichier de
  statistiques, `test` et `predict` refusent seulement quand le pipeline
  demandé en a besoin.
