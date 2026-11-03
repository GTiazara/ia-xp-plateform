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
│   └── perceptron.py            # premier modèle
└── data/
    ├── dataset/
    │   ├── base.py              # x et y, rien d'autre
    │   └── table_dataset.py
    └── datamodule/
        └── table_datamodule.py  # lecture CSV, Feather, HDF5
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
  `Perceptron`, et les fonctions internes de
  `TableDataModule` (lecture de fichier, découpage automatique, jointure
  par `id_column`, et chaque cas d'erreur : id manquant, dupliqué, colonne
  absente, valeur de split inconnue).
- `test/integration/` couvre `TableDataModule` de bout en bout (fichier
  réel sur disque, 3 formats, les deux modes de split) et l'enchaînement
  dataset/datamodule → `DataLoader` → modèle.
