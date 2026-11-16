# Tasks

## 1. Semis de l'entraînement

- [x] 1.1 En première ligne de `run_train`, appeler `pytorch_lightning.seed_everything(int(args.seed), workers=True, verbose=False)` avant `build_datamodule`. Ne pas semer `run_test` ni `run_predict`. Ne pas passer de générateur au `DataLoader`. Dans `test/unitaire/test_table_datamodule_helpers.py`, le même tableau découpé deux fois avec le même `split_seed`, après deux `torch.manual_seed` différents, donne les mêmes identifiants de validation. Dans `test/unitaire/test_cli.py`, l'aide de `train` contient « initialisation ». Lancer `poetry run pytest -q test/unitaire/test_table_datamodule_helpers.py test/unitaire/test_cli.py`.

- [x] 1.2 Ajouter `test/integration/test_seeded_training.py`. Deux `ia-xp train` sur CPU, graine `7`, une époque, le même tableau, deux dossiers de sortie : les tenseurs `model.*` des checkpoints sont identiques. Un troisième entraînement, graine `8`, diffère d'au moins un poids. `--accelerator cpu` est explicite. Lancer `poetry run pytest -q test/integration/test_seeded_training.py test/integration/test_cli_flow.py`.

## 2. Documentation

- [x] 2.1 Dans le README, dire que la graine de la commande d'entraînement sème aussi l'initialisation et l'ordre des batches, et que le découpage garde les mêmes lignes. Les phrases ajoutées ne citent aucun nombre de `conf/`. Dans `conf/config.yaml`, mettre à jour seulement le commentaire de `seed` ; la valeur reste celle du fichier. Vérifier que le README contient « initialisation » et que `seed:` n'a pas changé de valeur.

## 3. Suite

- [x] 3.1 Lancer `poetry run pytest -q` et vérifier que la suite passe, sans nouvelle dépendance et sans fichier `checkpoints/` ni `predictions.csv` laissé par les tests dans le dépôt.
