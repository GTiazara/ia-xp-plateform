# Design

## Context

Voir `proposal.md` (Why). État du code observé :

- `TableDataModule.setup` lit le fichier, découpe en train/val/test, puis
  construit un `TableDataset` par split. Le dataset convertit les colonnes
  en tenseurs `float32` et `__getitem__` renvoie `(x[i], y[i])`.
- `predict` ne passe pas par le datamodule : il lit le fichier, appelle
  `inputs_of` pour obtenir un tenseur, puis `Predictor.predict_proba`.
  `Predictor` n'accepte que des tenseurs et doit le rester.
- `test` reconstruit un datamodule complet puis appelle
  `trainer.test(..., ckpt_path=...)`.
- Les checkpoints ne contiennent que les poids du modèle
  (`state_dict` sous le préfixe `model.`). Le même dossier peut contenir
  `best-model.ckpt`, `best-model-v1.ckpt`, ...
- Les objets sont construits par des factories (nom → classe). Hydra
  compose des groupes ; `namespace_from_config` aplatit en un namespace
  que lisent les commandes. Pas d'instanciation Hydra (`_target_`).
- `openspec/specs/` est vide : tout est nouveau.

## Goals / Non-Goals

**Goals:**
- Un contrat de transformation qui ne suppose ni colonnes ni tableau,
  pour que des données raster le réutilisent.
- Statistiques apprises sur l'entraînement, rejouées à l'identique au
  test et à la prédiction, avec le bon checkpoint.
- Comportement inchangé quand aucune transformation n'est demandée.

**Non-Goals:**
- Transformations raster : seul le contrat est pensé pour elles.
- Mise à l'échelle de la cible, encodage de variables catégorielles.
- Option argparse pour `ia-xp`.
- Choisir le groupe par défaut `standardize` : l'utilisateur l'active.

## Decisions

### Package `transform/`, pas de dépendance vers `factory/`

`ia_xp_plateform/transform/` contient le contrat (`base.py`), les
transformations (`scaling.py` : `standardize`, `min_max` ;
`elementwise.py` : `clip`, `gaussian_noise`) et l'assemblage
(`pipeline.py`). La construction par nom vit dans
`factory/transform_factory.py` (`TRANSFORMS`, `create_transform`,
`create_preprocessing`), comme pour les modèles et les loggers. Le
datamodule importe `transform/`, jamais `factory/` : il reçoit un
objet déjà construit.

Le dossier s'appelle `transform` (singulier, comme `metric`, `module`).
Le groupe Hydra s'appelle `augmentation`, selon la demande ; les deux
noms ne se confondent pas : le groupe décrit *quoi* appliquer, le package
décrit *comment*.

### Contrat : classe simple, tenseur dans, tenseur sorti

```python
class BaseTransform:
    name: str                       # clé de la factory ET clé d'appariement
    stateful: bool = False          # a-t-elle des statistiques à apprendre ?
    def fit(self, x) -> "BaseTransform": ...   # x = tout le train
    def __call__(self, x): ...                 # un exemple ou un lot
    def state_dict(self) -> dict: ...          # nombres et listes seulement
    def load_state_dict(self, state: dict): ...
```

Pas de sous-classe de `nn.Module` : l'état est du JSON, pas des buffers,
et rien ici ne doit voyager vers le GPU (les transformations tournent
dans le `Dataset`, sur CPU). `standardize` et `min_max` calculent leurs
statistiques sur `dim=0` ; elles s'appliquent par broadcast à un exemple
`(f,)` comme à un lot `(n, f)`. Une transformation raster future
calculera ses propres statistiques (par canal) derrière le même contrat.

Alternative écartée : `torchvision.transforms`. Dépendance lourde,
pensée pour des images, sans notion d'état appris exportable.

Alternative écartée : transformer dans le datamodule, une fois pour
toutes, sur le tableau pandas. Cela interdit le bruit tiré à chaque
époque, et lie le prétraitement au tableau.

### Application dans `BaseDataset.__getitem__`

`BaseDataset(x, y, transform=None)` applique `transform` à `x[i]` quand
il existe. Les tenseurs bruts restent dans `dataset.x`. Une
transformation aléatoire est donc retirée à chaque accès, donc à chaque
époque, sans état global. Coût : un appel Python par exemple ; pour
quelques dizaines de milliers de lignes tabulaires, négligeable.

Alternative écartée : transformer par lot dans un `collate_fn`. Plus
rapide, mais le contrat « un exemple ou un lot » n'y gagne rien et le
test d'un exemple isolé devient indirect.

### Quatre pipelines, une classe `Preprocessing`

`Preprocessing` possède quatre `Pipeline` (`train`, `val`, `test`,
`predict`). Un `Pipeline` est une liste de transformations appliquées
dans l'ordre ; vide, il renvoie son entrée.

`Preprocessing.fit(train_x)` :
1. passe `train_x` dans le pipeline `train`, étape par étape : chaque
   étape à statistiques apprend sur la sortie de la précédente (si
   `clip` précède `standardize`, la moyenne vient des valeurs bornées ;
   si `gaussian_noise` précède, elle vient des valeurs bruitées, ce
   qu'un modèle verra vraiment) ;
2. récupère la liste d'états `[{"name", "state"}, ...]` des étapes à
   statistiques du pipeline `train` ;
3. charge ces états dans `val`, `test` et `predict`.

`Preprocessing.load_states(states)` fait seulement l'étape 3 pour les
quatre pipelines, `train` compris. C'est ce que `test` et `predict`
appellent avec le fichier JSON relu. Un attribut `fitted` évite de
réapprendre quand `load_states` a déjà été appelé.

Ni `val`, ni `test`, ni `predict` n'apprennent jamais : c'est la règle
« pas de fuite ».

### Appariement par `name`, n-ième occurrence

Une étape à statistiques dans un autre pipeline reprend l'état de
l'étape de même `name` dans la liste d'états, en comptant les
occurrences (la deuxième `standardize` de `val` prend la deuxième de
`train`). Sans équivalent : `ValueError` qui nomme la transformation et
le split.

Alternatives écartées :
- *Apparier par position* : insérer un `clip` devant ferait décaler
  tous les états sans erreur.
- *Donner un `id` explicite à chaque étape* : plus sûr, mais alourdit
  chaque YAML pour un cas (deux `standardize` dans un même pipeline) qui
  n'existe pas dans les besoins actuels.
- *Réapprendre dans chaque pipeline* : fuite des statistiques de `val`
  et `test`, et `predict` n'a pas de jeu d'apprentissage.

### État écrit à côté du checkpoint, sous le même nom

`run_train` écrit, après `fit`, `Path(best_model_path).with_suffix(
".preprocessing.json")` : `best-model.preprocessing.json` à côté de
`best-model.ckpt`. `test` et `predict` calculent le même chemin depuis
`checkpoint=`. Le fichier est toujours écrit (liste vide sans
transformation à statistiques), ce qui fait de sa présence une
vérification simple. Si le dossier des checkpoints ne produit aucun
meilleur checkpoint (`best_model_path` vide), rien n'est écrit.

Alternatives écartées :
- *Un seul `preprocessing.json` par dossier* : un second entraînement
  (`best-model-v1.ckpt`) l'écraserait, et l'ancien checkpoint
  prédirait avec de mauvaises statistiques, sans erreur.
- *Mettre l'état dans le `.ckpt`* : le `ClassificationModule` ne connaît
  pas le prétraitement ; `Predictor` relit seulement `model.*`. Il
  faudrait toucher le module et le chargement pour un gain faible.
- *Recalculer sur le fichier de données au test* : refaire l'apprentissage
  suppose le même découpage et les mêmes données, et `predict` n'a pas
  de découpage.

### Conf : groupe `augmentation`, défaut `none`

```yaml
# conf/augmentation/standardize.yaml
train:   [{name: standardize}]
val:     [{name: standardize}]
test:    [{name: standardize}]
predict: [{name: standardize}]
```

`none.yaml` met les quatre listes à `[]`. `conf/config.yaml`,
`conf/train_mlp.yaml` et `conf/train_perceptron.yaml` ajoutent
`- augmentation: none` à leurs `defaults`. Activation :
`poetry run ia-xp-conf augmentation=standardize`.

`namespace_from_config` ajoute `args.augmentation`, un dict simple
(`OmegaConf.to_container(..., resolve=True)`), `{}` si le groupe est
absent. `common.build_datamodule` et `predict` appellent
`create_preprocessing(getattr(args, "augmentation", None))` ; le CLI
argparse n'a pas cet attribut, donc il ne prétraite pas.

Pourquoi `none` par défaut : `standardize` changerait les résultats de
toute expérience existante et les valeurs de `val_loss` déjà comparées.
L'utilisateur choisit quand basculer.

### Erreurs claires, tôt

- Nom inconnu ou paramètre inconnu : `ValueError` à la construction
  (donc avant de lire un fichier de données), qui liste les noms
  disponibles.
- `standardize` appliquée sans `fit` ni état : `RuntimeError`.
- État manquant ou fichier absent : `FileNotFoundError` / `ValueError`
  qui nomme le fichier et la transformation.

## Risks / Trade-offs

- [Un pipeline `val` contient une transformation à statistiques absente
  de `train`] → erreur explicite dès `setup`, jamais un défaut silencieux.
- [`gaussian_noise` placé dans `val` ou `test`] → rien ne l'interdit :
  une métrique bruitée est un choix de l'utilisateur. Le README le
  déconseille ; pas de règle codée.
- [Fichier d'état perdu ou copié sans le checkpoint] → `test` et
  `predict` échouent en nommant le fichier attendu, au lieu de prédire
  avec des entrées non normalisées.
- [Statistiques en float32 sur de très grands entiers] → écart minime
  par rapport à float64 ; assumé tant que les données sont déjà
  converties en `float32` par `TableDataset`.
- [Un appel Python par exemple ralentit le `DataLoader`] → mesuré
  négligeable à cette taille ; `num_workers` reste réglable.
- [`mean` / `std` d'une colonne à variance nulle] → l'écart-type 0 est
  remplacé par 1 (idem pour l'étendue de `min_max`) : la colonne devient
  constante 0, sans `NaN`.
- [Les valeurs de validation hors de la plage apprise dépassent `[0, 1]`
  avec `min_max`] → comportement voulu et documenté (pas de troncature
  silencieuse) ; `clip` existe pour la demander.

## Migration Plan

Aucun déploiement. Les expériences existantes ne changent pas (`none`
par défaut) ; `ia-xp` (argparse) non plus. Un checkpoint écrit avant
cette modification reste utilisable tant qu'on n'active pas de
transformation à statistiques. Retour arrière : retirer
`- augmentation: none` des `defaults` ; le reste du code ignore les
pipelines vides.
