# Design

## Context

Voir `proposal.md` pour le motif. Le seul fichier concerné est
`test/unitaire/test_configuration.py`. Il charge les YAML réels de
`conf/` via `load_config` et `namespace_from_config`. OmegaConf résout
les interpolations (`${seed}`) et fusionne les groupes avec `_self_`.

Les littéraux numériques aujourd'hui figés, et leur source :

| Assertion | Source dans `conf/` |
| --- | --- |
| `model.input_features == 1` | `config.yaml` (`model.input_features`, actuellement 16) |
| `model.output_features == 1` | `model/perceptron.yaml` |
| `trainer.max_epochs == 200` et `args.epochs == 200` | `trainer/default.yaml` |
| `module.learning_rate == 0.01` et `args.learning_rate == 0.01` | `module/classification.yaml` |
| `callbacks.early_stopping_patience == 5` et `args.patience == 5` | `callbacks/default.yaml` |
| `hidden_features == 16`, `hidden_layers == 1` | `model/mlp.yaml`, écrasé par `train_mlp.yaml` (actuellement 10 couches) |

## Goals / Non-Goals

**Goals:**

- Un changement de nombre dans un YAML d'expérience ne fait plus
  échouer `test_configuration.py`.
- Le test continue de prouver la composition : groupes choisis, types,
  contraintes, interpolation, et copie fidèle vers le namespace du CLI.

**Non-Goals:**

- Introduire un second jeu de YAML figé pour les tests.
- Modifier `conf/`, `load_config` ou `namespace_from_config`.
- Relâcher les assertions non numériques (noms de groupes, `mode`,
  `target_columns`, `accelerator`, chemins de logs, `input_features is
  None` dans `train_perceptron.yaml`).

## Decisions

### Contrainte de type, puis égalité config → namespace

Pour chaque champ numérique lu depuis un YAML d'expérience, le test
affirme le type et la contrainte, puis, quand le namespace existe, que
la commande a recopié la valeur composée.

```python
assert isinstance(cfg.trainer.max_epochs, int) and cfg.trainer.max_epochs >= 1
assert args.epochs == cfg.trainer.max_epochs
```

`learning_rate` est un flottant strictement positif (`int` exclu, pour
ne pas accepter un `1` posé par erreur à la place de `0.01` sans que
le type ait changé — en pratique OmegaConf rend `0.01` en `float`).
`input_features` accepte `None` ou un entier `>= 1` : `null` dans le
YAML veut dire « la commande compte les colonnes ».

Alternative écartée : copier `conf/` vers une fixture aux nombres
figés. Elle protège les tests, mais elle ne compose plus les fichiers
que l'utilisateur lance vraiment, et elle double la maintenance.

Alternative écartée : ne plus charger `conf/` du tout et ne tester que
des surcharges. On perdrait la preuve que les groupes par défaut
(`data=perceptron`, `model=perceptron`, `logger=tensorboard`) se
assemblent.

### `train_perceptron` garde `input_features is None`

Ce `None` n'est pas un hyperparamètre. `conf/train_perceptron.yaml`
documente que la commande déduit le nombre de colonnes. Le test
continue de l'exiger. `train_mlp.yaml`, lui, fixe un entier : le test
vérifie seulement que c'est un entier `>= 1`, et que
`hidden_layers` / `hidden_features` le sont aussi.

### Les surcharges du test restent des égalités exactes

`test_override_replaces_a_value_and_the_interpolation_follows` envoie
`seed=7` et `trainer.max_epochs=3`. Ces nombres sont le stimulus du
test, pas un réglage d'expérience. Ils restent en égalité stricte, y
compris `cfg.data.split_seed == 7`.

### Aide locale au test, pas un utilitaire partagé

Deux petites fonctions dans `test_configuration.py`
(`assert_positive_int`, `assert_positive_float`) suffisent. Les autres
fichiers de test construisent leurs objets avec leurs propres nombres ;
ils n'ont pas ce problème.

## Risks / Trade-offs

- [Un YAML peut poser `max_epochs: 0` et le test le rejette, alors que Hydra l'accepterait] → la contrainte `>= 1` est celle déjà imposée par l'entraînement (`min_epochs: 1`) et par `early_stopping_patience`. Un zéro est une config invalide pour cette application.
- [Les chemins et `accelerator: gpu` restent figés] → hors du périmètre demandé (valeurs numériques). Un renommage de `logs/xp01` cassera encore le test ; c'est assumé.
- [`train_mlp.yaml` peut repasser `hidden_layers` à une valeur absurde comme `-1`] → le test d'entier `>= 1` l'attrape. Il n'attrape pas une profondeur déraisonnable (10, 100) : c'est voulu, c'est un réglage d'expérience.

## Migration Plan

Aucun déploiement. Après modification du fichier de test,
`poetry run pytest -q test/unitaire/test_configuration.py` doit passer
sans toucher aux YAML, puis la suite complète.

Retour arrière : restaurer les égalités littérales dans le fichier de
test. Aucune donnée ni checkpoint n'est impliqué.
