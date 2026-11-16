# Tasks

## 1. Tests de configuration

- [x] 1.1 Dans `test/unitaire/test_configuration.py`, remplacer les égalités numériques lues depuis `conf/` par une contrainte de type, et vérifier que le namespace recopie la valeur composée. Concerne `test_default_config_selects_the_perceptron_experiment` (`input_features`, `output_features`, `max_epochs`, `learning_rate`, `early_stopping_patience`), `test_mlp_group_selects_the_hidden_layer_model` (`output_features`, `hidden_features`, `hidden_layers`), `test_named_configs_choose_the_model_and_their_own_log_directory` (`hidden_layers`, `hidden_features`, `input_features` du MLP) et `test_namespace_flattens_the_groups_for_the_commands` (`epochs`, `patience`, `learning_rate`). Garder les égalités des surcharges posées par le test (`seed=7`, `max_epochs=3`) et `input_features is None` sur `train_perceptron`. Vérifier avec `poetry run pytest -q test/unitaire/test_configuration.py`.
- [x] 1.2 Ajouter une phrase dans la section Tests du README : la composition Hydra est vérifiée sans figer les nombres des fichiers d'expérience. Vérifier que la phrase est présente et qu'elle ne cite aucun nombre de `conf/`.
- [x] 1.3 Lancer `poetry run pytest -q` et vérifier que les deux échecs de `test_configuration.py` ont disparu, sans modifier les YAML de `conf/`.
