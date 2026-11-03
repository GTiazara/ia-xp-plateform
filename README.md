# ia-xp-plateform

Plateforme d'expériences (**XP**) pour apprendre l'intelligence artificielle
pas à pas : chaque XP est un modèle, une tâche d'apprentissage et un jeu de
données autonomes, rangés dans une architecture modulaire inspirée de
[`bdf-model`](https://gitlab.ign.fr/bdforet/bdf-model) (le pipeline
PyTorch Lightning + Hydra utilisé pour BDForet), mais réduite à l'essentiel.

## Architecture

| `bdf-model`                         | `ia-xp-plateform`                      | Rôle                                            |
| ------------------------------------ | --------------------------------------- | ------------------------------------------------ |
| `bdf/train/models/`                 | `xp/train/models/`                     | Modèles (ex: XP-01 → `Perceptron`)              |
| `bdf/train/module/classification.py`| `xp/train/module/binary_classification.py` | Tâche d'apprentissage (modèle + perte + optim + métriques) |
| `bdf/train/data/`                   | `xp/train/data/`                       | Datasets + `LightningDataModule`                |
| `bdf/cli/launch_experiment.py`      | `xp/cli/launch_experiment.py`          | Orchestration d'une XP (setup → fit/test/predict) |
| `bdf/main.py`                       | `xp/main.py`                           | Point d'entrée Hydra                            |
| `conf/`                             | `conf/`                                | Configuration modulaire (`task`, `datamodule`, `trainer`, `callbacks`, `logger`, `paths`) |

Chaque composant (modèle, optimiseur, perte, datamodule, trainer, callbacks,
logger) est instancié par [Hydra](https://hydra.cc/) à partir de fichiers de
configuration sous `conf/`, exactement comme dans `bdf-model`. On peut donc
faire évoluer ou remplacer un composant (par ex. passer du perceptron à un
MLP) sans toucher au reste de l'architecture.

## XP disponibles

- **[XP-01 — Perceptron / décision d'accélérer](docs/xp-01-perceptron-conduite.md)** :
  un perceptron apprend à décider s'il faut accélérer ou non, à partir de 4
  entrées simples (distance à l'obstacle, présence d'un obstacle, position
  de la voiture, vitesse), sur un jeu de données synthétique. Sert de
  validation de l'architecture avant de passer à des modèles et des
  problèmes plus complexes (MLP, CNN, apprentissage par renforcement).

## Démarrage rapide

```bash
poetry install
cp .env.dist .env   # optionnel, LOG_DIR vaut "./logs" par défaut

# Lancer l'XP-01
poetry run python xp/main.py
# ou
./run.sh

# Lancer les tests
poetry run pytest
```

## Structure du dépôt

```
ia-xp-plateform/
├── xp/                          # package Python de la plateforme
│   ├── main.py                  # point d'entrée Hydra
│   ├── cli/                     # orchestration d'une XP
│   └── train/
│       ├── models/              # modèles (BaseModel, Perceptron, ...)
│       ├── module/              # tâches d'apprentissage (LightningModule)
│       └── data/                # datasets + datamodules
├── conf/                        # configuration Hydra (une sous-arborescence par composant)
├── docs/                        # fiche détaillée de chaque XP
├── tests/                       # tests unitaires
└── openspec/                    # suivi des changements (OpenSpec)
```
