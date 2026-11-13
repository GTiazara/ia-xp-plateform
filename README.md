# ia-xp-plateform

Plateforme pour apprendre l'intelligence artificielle pas à pas. Chaque
brique a un seul rôle : le dataset expose des exemples `(x, y)`, le modèle
calcule une sortie, le datamodule charge les fichiers. Un entraîneur futur
n'aura pas à connaître l'intérieur de ces briques.

Python 3.12. Les dépendances sont gérées par Poetry, et le projet est
installé comme paquet (`ia_xp_plateform`) dans un environnement virtuel
créé dans le dépôt.

```bash
poetry install
```

## Structure

```text
ia_xp_plateform/
├── models/
│   ├── base.py                  # contrat commun des modèles
│   ├── perceptron.py            # une couche linéaire
│   └── mlp.py                   # profondeur, largeur, N sorties
├── data/
│   ├── dataset/
│   │   ├── base.py              # x et y, rien d'autre
│   │   └── table_dataset.py
│   └── datamodule/
│       └── table_datamodule.py  # lecture CSV, Feather, HDF5
├── factory/
│   ├── model_factory.py         # nom -> BaseModel
│   ├── datamodule_factory.py    # nom -> LightningDataModule
│   └── logger_factory.py        # nom -> logger Lightning
├── callback/
│   └── callbacks.py             # checkpoint, arrêt anticipé, taux d'apprentissage
├── metric/
│   ├── binary_classification.py # justesse, précision, rappel, F1
│   └── confusion_matrix.py      # image de la matrice de confusion
├── module/
│   └── classification.py        # tâche de classification autour d'un BaseModel
├── inference/
│   └── predictor.py             # prédire avec un modèle entraîné, sans le fit
└── cli/
    ├── main.py                  # ia-xp train | test | predict
    ├── hydra_main.py            # ia-xp-conf, surcharges Hydra
    └── command/
```

## 1. Le dataset de base

`BaseDataset` reçoit deux tenseurs déjà prêts, `x` (entrées) et `y`
(cibles), et les expose comme un `torch.utils.data.Dataset` : `__len__`
donne le nombre d'exemples, `__getitem__` renvoie le couple `(x, y)` d'un
index. Il vérifie que `x` et `y` ont le même nombre de lignes.

Les datasets concrets n'ont qu'à construire ces deux tenseurs, puis appeler
`BaseDataset`.

## 2. Le dataset tabulaire

`TableDataset` part d'un tableau pandas déjà chargé. Il sépare les colonnes
en entrées et en cible, puis les convertit en tenseurs `float32`.

- `target_columns` : la colonne cible, ou une liste si `y` a plusieurs
  dimensions.
- `feature_columns` : les colonnes d'entrée. Par défaut, toutes les
  colonnes sauf la cible.

Il ne lit pas le fichier et ne nettoie pas le tableau. C'est le rôle du
datamodule.

```python
from ia_xp_plateform.data.dataset import TableDataset

dataset = TableDataset(dataframe, target_columns="y")
# feature_columns=["a", "b"] pour n'en garder que quelques-unes
```

## 3. Le contrat des modèles

`BaseModel` hérite de `torch.nn.Module` et rend `forward` obligatoire. Une
sous-classe qui oublie `forward` ne peut pas être instanciée.

Un entraîneur n'a donc besoin que de ce contrat : appeler `model(x)` et
récupérer un tenseur. Le nombre de couches, l'activation ou le nom de la
classe restent un détail du modèle. `parameters()`, `.to(device)` et
`state_dict()` restent disponibles, parce que `BaseModel` est un
`nn.Module`.

## 4. Le perceptron

`Perceptron` est un `BaseModel` à une seule couche linéaire (poids + biais).
`forward` calcule la somme pondérée, puis une sigmoïde la transforme en
probabilité entre 0 et 1.

```python
from ia_xp_plateform.models import Perceptron

model = Perceptron(input_features=2, output_features=1)
probabilite = model(x)  # (batch, 1)
```

## 4.1. Le MLP

`MLP`, dans `models/mlp.py`, empile `hidden_layers` couches cachées
(1 par défaut), chacune de largeur `hidden_features` (16 par défaut).
Chaque couche cachée
est une couche linéaire suivie d'une ReLU. La dernière couche, puis
une sigmoïde, donne `output_features` probabilités entre 0 et 1. Avec
`output_features=1`, `ClassificationModule` l'entraîne comme le
perceptron.

```python
from ia_xp_plateform.factory import create_model

model = create_model(
    "mlp",
    input_features=2,
    hidden_features=16,
    hidden_layers=2,
    output_features=1,
)
```

```bash
poetry run ia-xp-conf model=mlp model.hidden_features=16 model.hidden_layers=2 model.output_features=1
```

`model.input_features` doit rester égal au nombre de colonnes attributs.

## 5. Le datamodule tabulaire

`TableDataModule` est un `LightningDataModule`. C'est le seul endroit qui
lit un fichier. Le format est déduit de l'extension :

| Extension | Lecture |
| --- | --- |
| `.csv` | CSV |
| `.feather` | Feather |
| `.h5`, `.hdf5` | HDF5 (`hdf_key` si le fichier contient plusieurs tables) |

Les données restent dans **un seul fichier**. Chaque ligne porte un
identifiant unique (`id_column`, `"id"` par défaut). Cette colonne n'est
pas une entrée du modèle : elle sert à reconnaître l'exemple. Sans elle,
ou si une valeur est en double, le chargement échoue.

Le découpage train / val / test se fait de deux façons.

### Le datamodule découpe

Sans fichier de split, il mélange les lignes (`split_seed`, 42 par défaut)
puis en garde 20 % pour la validation (`val_split`) et 10 % pour le test
(`test_split`). Le reste va à l'entraînement. Le fichier doit quand même
contenir la colonne `id`.

```python
from ia_xp_plateform.data.datamodule import TableDataModule

datamodule = TableDataModule(
    data_path="data.csv",
    target_columns="y",
    batch_size=32,
)
datamodule.setup()
batch_x, batch_y = next(iter(datamodule.train_dataloader()))
```

### Un fichier `split` accompagne les données

Le fichier compagnon a le même format et le même dossier. Il ne copie pas
les données, et n'a pas besoin d'être dans le même ordre que `data` : le
rapprochement se fait par un identifiant (`id_column`, `"id"` par défaut),
présent dans les deux fichiers.

```text
data.csv          split.csv
data.feather      split.feather
data.h5           split.h5
```

`data.csv` :

```csv
id,a,b,y
0,1.2,3.4,0
1,5.6,7.8,1
2,9.0,1.1,0
```

`split.csv` (ordre libre) :

```csv
id,split
2,train
0,val
1,train
```

```python
datamodule = TableDataModule(
    data_path="data.csv",
    split_path="split.csv",
    target_columns="y",
)
```

Les valeurs acceptées pour `split` sont `train`, `val` et `test`. S'il n'y
a aucune ligne `val` ou `test`, le `DataLoader` correspondant vaut `None`.
Chaque identifiant de `data` doit avoir exactement une ligne dans `split`
(ni manquant, ni en trop, ni dupliqué), sinon une erreur explicite est
levée. `id_column` (`"id"` par défaut, personnalisable) est exclue des
colonnes d'entrée par défaut.

Pour un HDF5, `hdf_key` désigne la table de `data`, et `split_hdf_key`
celle de `split` (la même clé par défaut).

Dans les deux cas, les colonnes d'entrée sont choisies une fois sur le jeu
d'entraînement (toutes sauf la cible et `id_column`, sauf si
`feature_columns` est donné) puis réutilisées pour la validation et le
test.

## 6. Les factories

Une factory construit un objet à partir de son nom. Le code qui lance une
expérience n'importe ni `Perceptron` ni `TableDataModule` : il donne un
nom et des paramètres.

| Nom | Objet construit |
| --- | --- |
| `"perceptron"` | `Perceptron` |
| `"mlp"` | `MLP` (`hidden_layers` couches cachées, `output_features` sorties) |
| `"table"` | `TableDataModule` |
| `"tensorboard"` | `TensorBoardLogger` (logger par défaut) |
| `"litlogger"` | `LitLogger` |

```python
from ia_xp_plateform.factory import create_datamodule, create_model

model = create_model("perceptron", input_features=2, output_features=1)
datamodule = create_datamodule(
    "table",
    data_path="data.csv",
    target_columns="y",
)
```

Un nom absent du registre lève une erreur qui liste les noms connus.

## 7. Les callbacks

`create_callbacks` prépare les trois observateurs branchés sur
l'entraînement. Ils regardent tous la perte de validation `val_loss` :
le checkpoint garde le meilleur modèle, l'arrêt anticipé coupe si elle
ne baisse plus, et le moniteur note le taux d'apprentissage à chaque
époque. Les poids sont écrits dans `output_dir/checkpoints`.

```python
from ia_xp_plateform.callback import CallbackConfig, create_callbacks

callbacks = create_callbacks(
    CallbackConfig(output_dir="logs/xp01", early_stopping_patience=5)
)
```

`early_stopping_patience` vaut au moins 1. Le modèle entraîné doit
journaliser `val_loss` pour que le checkpoint et l'arrêt anticipé
aient quelque chose à surveiller. Le suivi du taux d'apprentissage
exige un logger sur le `Trainer`.

## 8. Le module de classification

Un `BaseModel` ne sait pas calculer une perte ni mettre ses poids à
jour. `ClassificationModule` porte cette tâche pour n'importe quel
`BaseModel` qui renvoie une probabilité entre 0 et 1 : perte `BCELoss`,
optimiseur Adam, et journalisation de `train_loss`, `val_loss`,
`test_loss`, `train_acc`, `val_acc`, `test_acc`, puis, sur l'époque
entière, `train_precision`, `val_precision`, `test_precision`,
`train_recall`, `val_recall`, `test_recall`, `train_f1`, `val_f1`,
`test_f1`. À chaque époque, le
module envoie aussi deux images à TensorBoard (onglet Images).
`train_confusion_matrix` et `val_confusion_matrix` sont les effectifs.
`train_confusion_matrix_normalized` et
`val_confusion_matrix_normalized` divisent chaque ligne par son total :
la part des prédictions dans chaque classe réelle. Une classe absente
reste à 0. Les lignes sont la classe réelle, les colonnes la classe
prédite.

Ces chiffres ne sont pas calculés ici. Le module appelle
`BinaryClassificationMetrics` (`ia_xp_plateform/metric`) : `reset` au
début de l'époque, `update` à chaque lot, `compute` à la fin. Un futur
module de régression branchera un autre objet de métriques. La perte
et l'optimiseur restent dans le module.

La précision est la part de vrais positifs parmi les exemples annoncés
positifs. Le rappel est la part des positifs réels qui ont été
retrouvés. Le F1 est leur moyenne harmonique : il baisse dès que l'un
des deux baisse, et vaut 0 quand il n'y a aucun vrai positif. La
précision et le rappel valent 0 quand leur dénominateur est nul
(aucun positif annoncé, ou aucun positif dans l'époque).

`val_loss` est le nom attendu par `create_callbacks` : le module et les
callbacks se branchent ensemble sans réglage supplémentaire.

```python
from ia_xp_plateform.factory import create_model
from ia_xp_plateform.module import ClassificationModule

model = create_model("perceptron", input_features=2, output_features=1)
module = ClassificationModule(model=model, learning_rate=1e-2)
```

Changer de modèle plus tard (un MLP par exemple) ne touche pas à
`ClassificationModule`.

## 9. L'inférence

Prédire avec un modèle entraîné ne demande ni `Trainer`, ni datamodule,
ni perte, ni optimiseur : seulement l'architecture (un `BaseModel`) et ses
poids. `Predictor` réunit les deux et se sert du checkpoint écrit pendant
l'entraînement (`output_dir/checkpoints/best-model.ckpt`).

```python
import torch
from ia_xp_plateform.inference import Predictor

predictor = Predictor.from_checkpoint(
    "logs/xp01/checkpoints/best-model.ckpt",
    model_name="perceptron",
    model_kwargs={"input_features": 2, "output_features": 1},
)

x = torch.tensor([[1.0, 2.0], [-1.0, -2.0]])
predictor.predict_proba(x)  # probabilités entre 0 et 1, forme (2, 1)
predictor.predict(x)        # classes 0 ou 1, forme (2, 1)
```

Le checkpoint ne contient que les poids : `model_name` et `model_kwargs`
doivent être ceux de l'entraînement. Si les poids ne correspondent pas à
l'architecture demandée, une erreur explicite est levée. `Predictor`
passe le modèle en mode évaluation, coupe les gradients, et applique le
même seuil que `ClassificationModule` (`threshold`, `0.5` par défaut).

Pour un modèle déjà construit, `Predictor(model, threshold=0.5)` suffit.
Les entrées sont des tenseurs `(batch, n_features)` ; les colonnes
doivent être dans le même ordre que pendant l'entraînement.

## 10. Le CLI

`ia-xp` enchaîne les briques sans écrire de Python. `--input-features`
peut être omis : il est déduit du nombre de colonnes attributs.

```bash
poetry run ia-xp train \
  --data data.csv --target y --output-dir logs/xp01 --epochs 10

poetry run ia-xp test \
  --data data.csv --target y \
  --checkpoint logs/xp01/checkpoints/best-model.ckpt

poetry run ia-xp predict \
  --data data.csv \
  --checkpoint logs/xp01/checkpoints/best-model.ckpt \
  --output predictions.csv
```

`train` affiche le chemin du checkpoint. `predict` écrit `id`,
`probability` et `prediction`. Sans `--features`, les entrées sont
toutes les colonnes sauf `id` et, si `--target` est donné, la cible.
Sur un fichier qui contient encore la colonne cible, il faut donc la
nommer pour qu'elle ne devienne pas une entrée. L'appareil est le CPU
(`--accelerator cpu`) tant qu'on ne demande pas autre chose.

La même commande existe sous `python -m ia_xp_plateform.cli`.

## 11. La configuration Hydra

`conf/` décrit l'expérience sans écrire de Python. Hydra assemble un
fichier par groupe. Les factories lisent les noms (`table`,
`perceptron`) et construisent les objets.

```text
conf/
├── config.yaml                  # expérience par défaut
├── train_perceptron.yaml        # poetry run ia-xp-conf train_perceptron
├── train_mlp.yaml               # poetry run ia-xp-conf train_mlp
├── data/
│   └── perceptron.yaml          # fichier tabulaire, cible, splits
├── model/
│   └── perceptron.yaml          # nom de factory, entrées, sorties
├── trainer/
│   └── default.yaml             # époques, appareil
├── module/
│   └── classification.yaml      # taux d'apprentissage, seuil
├── callbacks/
│   └── default.yaml             # patience de l'arrêt anticipé
└── logger/
    ├── tensorboard.yaml         # suivi par défaut, local
    └── litlogger.yaml           # suivi en ligne, lightning.ai
```

`data/perceptron.yaml` porte le jeu de données. Le datamodule
reste `name: table` : c'est la factory qui sait lire un CSV, un
Feather ou un HDF5. `split_seed` reprend `${seed}` du fichier choisi.

Chaque fichier `conf/*.yaml` est une expérience. Le premier argument
de `ia-xp-conf`, s'il ne contient pas `=`, est son nom. Sans argument,
on lance `config.yaml`. Les arguments suivants restent des surcharges.
`train_perceptron` et `train_mlp` écrivent dans `logs/train_perceptron`
et `logs/train_mlp`. Le MLP range ses courbes sous le nom `mlp`.

```bash
poetry run ia-xp-conf train_mlp
poetry run ia-xp-conf train_perceptron trainer.max_epochs=10

poetry run ia-xp-conf \
  data.data_path=data.csv trainer.max_epochs=10

poetry run ia-xp-conf mode=test \
  data.data_path=data.csv \
  checkpoint=logs/xp01/checkpoints/best-model.ckpt

poetry run ia-xp-conf mode=predict \
  data.data_path=data.csv \
  checkpoint=logs/xp01/checkpoints/best-model.ckpt \
  predictions=predictions.csv
```

Une surcharge ne modifie pas les fichiers. Choisir un autre fichier de
groupe plus tard s'écrit `model=autre` dès que `conf/model/autre.yaml`
existe. Les commandes `ia-xp train`, `test` et `predict` restent
disponibles pour un lancement sans fichier de configuration.

## 12. Le suivi de l'expérience

L'entraîneur n'écrit pas lui-même les courbes. `ClassificationModule`
émet `train_loss`, `val_loss`, `val_acc` et les autres métriques ;
`LearningRateMonitor` ajoute le taux d'apprentissage. Un logger les
reçoit. Le logger par défaut est TensorBoard, choisi par
`conf/logger/tensorboard.yaml` : local, sans compte.

TensorBoard range chaque lancement dans
`logs/tensorboard/<nom>/version_N`, avec N qui augmente à chaque run.
Les expériences passées restent donc à côté de la courante, et un seul
tableau de bord les compare :

```bash
poetry run ia-xp-conf data.data_path=data.csv model.input_features=1

poetry run tensorboard --logdir logs/tensorboard
```

Ouvre ensuite l'adresse affichée (`http://localhost:6006`) : les
courbes `train_loss`, `val_loss`, `val_acc`, `val_precision`,
`val_recall` et le taux d'apprentissage
se mettent à jour pendant l'entraînement. Pour classer un autre type
d'expérience à part, change le nom :
`logger.params.name=perceptron_bank`.

Avec `ia-xp train`, le logger vaut aussi `tensorboard` par défaut et
écrit dans `<output-dir>/logger/experience`.

LitLogger (lightning.ai) reste disponible : `logger=litlogger`. Il
demande une connexion lightning.ai au premier lancement et affiche
l'adresse de l'expérience en ligne.

Changer d'outil ne change pas l'entraînement. On ajoute le nom dans
`LOGGERS` et `LOGGER_DIR_KEYS` (`factory/logger_factory.py`), puis un
fichier de groupe, par exemple `conf/logger/clearml.yaml`, et on lance
avec `logger=clearml`. Le `Trainer` continue d'appeler le même logger.

## Tests

```bash
poetry run pytest
```

```text
test/
├── unitaire/      # chaque classe ou fonction testée isolément
└── integration/   # plusieurs briques ensemble (fichier sur disque,
                    # datamodule -> dataset -> DataLoader -> modèle)
```

- `test/unitaire/` couvre `BaseDataset`, `TableDataset`, `BaseModel`,
  `Perceptron`, `MLP`, `ClassificationModule`, `BinaryClassificationMetrics`
  (précision, rappel, F1, matrice de confusion), `Predictor` (checkpoints
  fabriqués à la main, chaque cas d'erreur), les factories
  (`create_model`, `create_datamodule`, `create_logger`), `create_callbacks`, et les
  fonctions internes de `TableDataModule` (lecture de fichier,
  découpage automatique, jointure par `id_column`, et chaque cas
  d'erreur : id manquant, dupliqué, colonne absente, valeur de split
  inconnue), le parseur du CLI, et la composition Hydra de `conf/`
  (groupes par défaut, interpolation `${seed}`, surcharge, groupe inconnu).
- `test/integration/` couvre `TableDataModule` de bout en bout (fichier
  réel sur disque, 3 formats, les deux modes de split), l'enchaînement
  factory → datamodule → `DataLoader` → modèle, un `Trainer.fit` complet
  avec `ClassificationModule` et les callbacks (y compris l'image de
  la matrice de confusion dans TensorBoard), la relecture du
  checkpoint obtenu par `Predictor`, le parcours `train` puis `test`
  puis `predict` du CLI, le même parcours lancé depuis `conf/`,
  les fichiers d'événements TensorBoard écrits par défaut, la
  conservation des expériences passées (`version_0`, `version_1`), et
  le choix de LitLogger par surcharge.
