# Design

## Context

Voir proposal.md pour le motif. `run_train` construit le datamodule, puis le modèle. `_split_dataframe` mélange les indices avec un `torch.Generator` privé semé par `split_seed`. `DataLoader(..., shuffle=True)` n'a pas de générateur : chaque époque tire sur le générateur global. `nn.Linear` tire ses poids à la construction, sur ce même générateur. `ia-xp` et `ia-xp-conf` appellent tous deux `run_train`. Hydra copie `cfg.data.split_seed` (`${seed}`) dans `args.seed`. L'argparse a `--accelerator` à `cpu` par défaut. Les tests d'intégration lisent la dernière ligne de stdout comme chemin de checkpoint.

## Goals / Non-Goals

**Goals:**

- Un seul appel, en tête de `run_train`, avant `build_datamodule` et `build_model`.
- Les deux entraînements CPU du spec partagent leurs poids. Le découpage ne bouge pas.

**Non-Goals:**

- `Trainer(deterministic=True)` ou `torch.use_deterministic_algorithms`.
- Semer `run_test`, `run_predict`, ou les `Trainer` construits directement dans les tests.
- Passer un `generator` au `DataLoader` : l'ordre des batches suit le générateur global semé par la commande. Le datamodule ne change pas.

## Decisions

1. `pytorch_lightning.seed_everything(int(args.seed), workers=True, verbose=False)` est la première ligne de `run_train`. L'import reste `pytorch_lightning`. `torch.manual_seed` seul laisserait NumPy, `random` et les workers de côté. Semer dans `hydra_main` ou `main` sèmerait aussi `test` et `predict`.
2. `workers=True` pose `PL_SEED_WORKERS`. Lightning applique alors `worker_init_fn` aux DataLoader du Trainer quand `num_workers` est supérieur à 0. Aujourd'hui ce nombre vaut 0, donc l'effet est nul. `workers=False` oublierait ces workers le jour où le nombre augmente.
3. `verbose=False`. Le message anglais de Lightning n'est pas le contrat du CLI. Les tests qui lisent la dernière ligne de stdout restent valides dans les deux cas.
4. Le test d'intégration passe par `ia-xp train` (`main`), `--accelerator cpu`, `--epochs 1`, `--seed 7` deux fois dans deux dossiers, puis `--seed 8`. Il charge les checkpoints et compare les tenseurs `model.*`. Le perceptron par défaut et le jeu binaire de la fixture suffisent. Un test du datamodule sème le générateur global avec deux valeurs, découpe avec le même `split_seed`, et compare les `id` de validation.

## Risks / Trade-offs

- [GPU non déterministe] → le spec et le test exigent CPU. Deux courbes GPU peuvent encore différer.
- [`PL_SEED_WORKERS` reste posé dans le processus] → les tests suivants ont `num_workers` à 0. Aucune restauration d'environnement.
- [Un dropout futur consomme le générateur CPU entre deux époques] → deux entraînements identiques restent égaux entre eux. On ne fige pas un générateur séparé dans le DataLoader.
- [Un ancien checkpoint non semé ne se reproduira pas] → les fichiers déjà écrits restent chargeables. Seuls les nouveaux entraînements suivent la graine.

## Migration Plan

Aucun. Pas de changement de format de checkpoint ni de fichier de split.
