# Spec Delta

## Purpose

Décrire dans `conf/` les transformations d'entrée de chaque split (train, val, test, predict), les apprendre sur l'entraînement seul, et les rejouer à l'identique au test et à la prédiction.

## ADDED Requirements

### Requirement: Le groupe `augmentation` décrit un pipeline par split
Un fichier du groupe `conf/augmentation/` SHALL pouvoir contenir quatre listes, `train`, `val`, `test` et `predict`, chacune faite de transformations données par `name` et leurs paramètres, appliquées dans l'ordre de la liste.

#### Scenario: Listes distinctes par split
- **WHEN** `train` contient `gaussian_noise` et `standardize`, et `val` contient seulement `standardize`
- **THEN** l'entraînement applique les deux, et la validation applique seulement `standardize`

#### Scenario: Liste absente
- **WHEN** un fichier du groupe ne définit pas `test`
- **THEN** le split de test n'est pas transformé

#### Scenario: Ordre des transformations
- **WHEN** une liste contient `clip` puis `standardize`
- **THEN** `standardize` apprend ses statistiques sur les valeurs déjà bornées

### Requirement: Sans transformation, le comportement actuel est conservé
Le fichier `none` du groupe SHALL laisser les quatre listes vides, et SHALL être le choix par défaut de `conf/config.yaml`.

#### Scenario: Lancement par défaut
- **WHEN** l'entraînement est lancé sans choisir de groupe `augmentation`
- **THEN** les entrées du modèle sont celles du fichier, sans modification

#### Scenario: Choix en ligne de commande
- **WHEN** l'utilisateur ajoute `augmentation=standardize` à la commande
- **THEN** les colonnes d'entrée sont standardisées pour tous les splits listés dans ce fichier

### Requirement: Les statistiques viennent du jeu d'entraînement seul
Les transformations à statistiques SHALL apprendre sur les lignes du split d'entraînement, et jamais sur celles de `val` ou de `test`.

#### Scenario: Pas de fuite
- **WHEN** le split de validation contient des valeurs extrêmes absentes de l'entraînement
- **THEN** les statistiques apprises sont celles de l'entraînement uniquement

### Requirement: Les autres splits réutilisent l'état appris par nom
Une transformation à statistiques listée dans `val`, `test` ou `predict` SHALL reprendre l'état de la transformation de même `name` du pipeline `train`.

#### Scenario: Même état partout
- **WHEN** `standardize` figure dans `train` et dans `val`
- **THEN** les deux utilisent la même moyenne et le même écart-type

#### Scenario: Pas d'équivalent dans train
- **WHEN** `standardize` figure dans `val` mais pas dans `train`
- **THEN** la préparation des données échoue avec une erreur qui nomme la transformation et le split

### Requirement: L'identifiant et la cible ne sont jamais transformés
Les pipelines SHALL s'appliquer aux colonnes d'entrée seulement, jamais à la colonne identifiant ni à la cible.

#### Scenario: Cible intacte
- **WHEN** `standardize` est active
- **THEN** les valeurs de la cible vues par la perte sont celles du fichier

### Requirement: L'entraînement écrit l'état appris à côté du checkpoint
À la fin de `train`, le système SHALL écrire, à côté du meilleur checkpoint et sous le même nom, un fichier lisible `<nom-du-checkpoint>.preprocessing.json` qui contient l'état appris de chaque transformation à statistiques du pipeline `train`.

#### Scenario: Fichier écrit
- **WHEN** `train` se termine avec `augmentation=standardize`
- **THEN** un fichier `best-model.preprocessing.json` existe à côté de `best-model.ckpt` et contient une moyenne et un écart-type par colonne d'entrée

#### Scenario: Aucune transformation à statistiques
- **WHEN** `train` se termine avec `augmentation=none`
- **THEN** le fichier est écrit avec une liste vide, sans erreur

#### Scenario: Plusieurs entraînements dans le même dossier
- **WHEN** un second entraînement produit `best-model-v1.ckpt` dans le même dossier
- **THEN** son état est écrit dans `best-model-v1.preprocessing.json` et celui du premier n'est pas modifié

### Requirement: Test et predict rejouent l'état écrit à l'entraînement
`test` et `predict` SHALL lire le fichier `<nom-du-checkpoint>.preprocessing.json` voisin du checkpoint donné par `checkpoint=` et SHALL l'appliquer à leur pipeline sans rien réapprendre.

#### Scenario: Prédiction cohérente
- **WHEN** `predict` s'exécute avec le checkpoint d'un entraînement standardisé
- **THEN** les lignes à prédire sont standardisées avec la moyenne et l'écart-type de l'entraînement

#### Scenario: État absent
- **WHEN** le pipeline demandé contient une transformation à statistiques et que le fichier d'état est introuvable
- **THEN** la commande échoue en nommant le fichier attendu

#### Scenario: Nom absent du fichier
- **WHEN** le pipeline demande `min_max` mais le fichier ne contient que `standardize`
- **THEN** la commande échoue en nommant la transformation manquante

### Requirement: Le CLI argparse reste inchangé
La commande `ia-xp` SHALL continuer à fonctionner sans prétraitement ; seules les commandes lancées depuis `conf/` utilisent le groupe `augmentation`.

#### Scenario: Aucun pipeline en argparse
- **WHEN** `ia-xp train` est lancé comme avant cette modification
- **THEN** l'entraînement se déroule sans erreur et sans transformation
