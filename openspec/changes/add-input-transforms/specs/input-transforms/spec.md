# Spec Delta

## Purpose

Prétraiter les entrées d'un modèle avec des transformations nommées, dont certaines apprennent leurs statistiques sur le jeu d'entraînement, sans dépendre du format des données.

## ADDED Requirements

### Requirement: Une transformation s'applique à un exemple comme à un lot
Chaque transformation SHALL renvoyer un tenseur de même forme que son entrée, qu'elle reçoive un seul exemple ou un lot d'exemples.

#### Scenario: Exemple isolé et lot donnent le même résultat
- **WHEN** une transformation déterministe est appliquée à un lot, puis à chacun de ses exemples
- **THEN** les résultats ligne à ligne sont identiques

#### Scenario: La forme est conservée
- **WHEN** une transformation est appliquée à un tenseur de forme `(n, f)`
- **THEN** le résultat a la forme `(n, f)`

### Requirement: Les transformations disponibles sont choisies par leur nom
Le système SHALL construire une transformation à partir de son nom et de ses paramètres, et SHALL proposer `standardize`, `min_max`, `clip` et `gaussian_noise`.

#### Scenario: Nom connu
- **WHEN** on demande la transformation `standardize`
- **THEN** une transformation prête à être apprise est renvoyée

#### Scenario: Nom inconnu
- **WHEN** on demande une transformation `inconnue`
- **THEN** une erreur cite ce nom et liste les transformations disponibles

#### Scenario: Paramètre invalide
- **WHEN** on demande `gaussian_noise` avec un écart-type négatif ou nul
- **THEN** une erreur nomme la transformation et le paramètre refusé

### Requirement: La standardisation centre et réduit chaque colonne
La transformation `standardize` SHALL apprendre la moyenne et l'écart-type de chaque colonne, puis renvoyer `(x - moyenne) / écart-type`.

#### Scenario: Colonnes d'échelles différentes
- **WHEN** `standardize` est apprise sur un tableau dont une colonne va de 0 à 5000 et une autre de 0 à 3
- **THEN** chaque colonne transformée du même tableau a une moyenne de 0 et un écart-type de 1

#### Scenario: Colonne constante
- **WHEN** une colonne apprise a le même nombre partout
- **THEN** sa valeur transformée vaut 0 et aucune division par zéro n'est produite

### Requirement: La mise à l'échelle min-max ramène chaque colonne dans [0, 1]
La transformation `min_max` SHALL apprendre le minimum et le maximum de chaque colonne, puis renvoyer `(x - minimum) / (maximum - minimum)`.

#### Scenario: Données d'apprentissage
- **WHEN** `min_max` est apprise sur un tableau puis appliquée à ce même tableau
- **THEN** chaque colonne a pour minimum 0 et pour maximum 1

#### Scenario: Valeur hors de la plage apprise
- **WHEN** `min_max` est appliquée à une valeur plus grande que le maximum appris
- **THEN** le résultat dépasse 1 et n'est pas tronqué

#### Scenario: Colonne constante
- **WHEN** une colonne apprise a le même nombre partout
- **THEN** sa valeur transformée vaut 0 et aucune division par zéro n'est produite

### Requirement: Le bornage limite les valeurs sans rien apprendre
La transformation `clip` SHALL ramener chaque valeur dans l'intervalle donné par ses paramètres `min` et `max`, dont au moins un est obligatoire.

#### Scenario: Valeurs hors de l'intervalle
- **WHEN** `clip` avec `min=0` et `max=10` reçoit `-5`, `3` et `40`
- **THEN** elle renvoie `0`, `3` et `10`

#### Scenario: Aucune borne
- **WHEN** `clip` est demandée sans `min` ni `max`
- **THEN** une erreur le signale

#### Scenario: Bornes inversées
- **WHEN** `clip` est demandée avec `min` plus grand que `max`
- **THEN** une erreur le signale

### Requirement: Le bruit gaussien ajoute un aléa centré
La transformation `gaussian_noise` SHALL ajouter à chaque valeur un nombre tiré d'une loi normale de moyenne 0 et d'écart-type donné.

#### Scenario: Deux appels, deux résultats
- **WHEN** `gaussian_noise` est appliquée deux fois à la même entrée
- **THEN** les deux résultats diffèrent

#### Scenario: Graine fixée
- **WHEN** la graine de PyTorch est fixée avant chacun de deux appels
- **THEN** les deux résultats sont identiques

### Requirement: Une transformation qui apprend doit être apprise avant d'être appliquée
Une transformation à statistiques (`standardize`, `min_max`) SHALL refuser de s'appliquer tant qu'elle n'a ni été apprise ni reçu un état.

#### Scenario: Application avant apprentissage
- **WHEN** `standardize` est appliquée sans apprentissage préalable
- **THEN** une erreur demande de l'apprendre ou de lui donner un état

### Requirement: L'état appris est exportable et rechargeable
Une transformation à statistiques SHALL exposer son état sous forme de données simples (nombres et listes), et SHALL se comporter de la même façon après rechargement de cet état.

#### Scenario: Aller-retour de l'état
- **WHEN** l'état d'une `standardize` apprise est exporté puis chargé dans une nouvelle `standardize`
- **THEN** les deux donnent le même résultat sur la même entrée

#### Scenario: État incompatible
- **WHEN** un état de `min_max` est chargé dans une `standardize`
- **THEN** une erreur le signale

#### Scenario: Transformation sans état
- **WHEN** l'état d'une `clip` est demandé
- **THEN** il est vide
