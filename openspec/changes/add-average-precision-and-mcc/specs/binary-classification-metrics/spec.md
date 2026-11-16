# Spec Delta

## Purpose

Mesurer, sur une époque de classification binaire, si les probabilités classent les positifs devant les négatifs, et si les quatre cases de la matrice s'accordent.

## ADDED Requirements

### Requirement: L'average precision ignore le seuil
L'average precision d'une époque SHALL être la moyenne des précisions calculées à chaque exemple positif, une fois les exemples triés par probabilité décroissante. Elle SHALL ignorer le seuil de décision.

#### Scenario: Classement parfait sous le seuil
- **WHEN** les probabilités sont `0.4`, `0.3`, `0.2`, `0.1`, les cibles `1`, `1`, `0`, `0`, et le seuil `0.5`
- **THEN** l'average precision vaut 1, et la précision, le rappel et le F1 valent 0

#### Scenario: Classement imparfait
- **WHEN** les probabilités sont `0.9`, `0.8`, `0.1` et les cibles `1`, `0`, `1`
- **THEN** l'average precision vaut `5/6`

#### Scenario: Scores égaux
- **WHEN** deux exemples ont la même probabilité et arrivent dans l'ordre cible `0` puis cible `1`
- **THEN** l'exemple arrivé en premier reste devant, et l'average precision vaut `1/2`

### Requirement: L'average precision vaut 0 sans positif
L'average precision SHALL valoir 0 lorsqu'aucun exemple de l'époque n'a la cible 1.

#### Scenario: Époque sans positif
- **WHEN** toutes les cibles valent 0
- **THEN** l'average precision vaut 0

#### Scenario: Époque vide
- **WHEN** aucun exemple n'a été compté
- **THEN** l'average precision vaut 0

### Requirement: L'average precision cumule les lots jusqu'à la remise à zéro
L'average precision SHALL être calculée sur tous les lots de l'époque, puis SHALL revenir à 0 après une remise à zéro.

#### Scenario: Deux lots
- **WHEN** un premier lot apporte la probabilité `0.9` et la cible `1`, puis un second la probabilité `0.1` et la cible `0`
- **THEN** l'average precision vaut 1

#### Scenario: Remise à zéro
- **WHEN** les compteurs sont remis à zéro après ces deux lots
- **THEN** l'average precision vaut 0

### Requirement: Le coefficient de Matthews utilise les quatre cases
Le coefficient de Matthews SHALL valoir `(VP * VN - FP * FN) / sqrt((VP + FP) * (VP + FN) * (VN + FP) * (VN + FN))`, avec les comptes obtenus au seuil de décision déjà utilisé pour la précision. Un dénominateur nul SHALL donner 0.

#### Scenario: Accord parfait
- **WHEN** il y a 2 vrais positifs, 2 vrais négatifs, et aucun faux
- **THEN** le coefficient vaut 1

#### Scenario: Désaccord parfait
- **WHEN** les probabilités sont `0.9`, `0.8`, `0.2`, `0.1`, les cibles `0`, `0`, `1`, `1`, et le seuil `0.5`
- **THEN** le coefficient vaut -1

#### Scenario: Matrice mélangée
- **WHEN** les probabilités sont `0.9`, `0.8`, `0.7`, `0.6`, `0.4`, `0.3`, `0.2` et les cibles `1`, `1`, `1`, `0`, `1`, `0`, `0`, au seuil `0.5`
- **THEN** le coefficient vaut `5/12`

#### Scenario: Prédiction toujours négative
- **WHEN** toutes les probabilités sont sous le seuil et au moins une cible vaut 1
- **THEN** le coefficient vaut 0

### Requirement: Chaque sortie est un exemple binaire
Plusieurs sorties d'un même exemple SHALL être mesurées comme autant d'exemples binaires indépendants, pour l'average precision comme pour le coefficient de Matthews.

#### Scenario: Deux sorties
- **WHEN** un exemple a les probabilités `0.9` et `0.1` et les cibles `1` et `0`
- **THEN** l'average precision vaut 1

### Requirement: Les deux mesures sont journalisées à chaque époque
L'entraînement SHALL publier l'average precision et le coefficient de Matthews de l'époque sous `train_ap`, `val_ap`, `test_ap`, `train_mcc`, `val_mcc` et `test_mcc`. L'average precision SHALL rester dans `[0, 1]` et le coefficient dans `[-1, 1]`.

#### Scenario: Fin d'époque de validation
- **WHEN** une époque de validation se termine
- **THEN** `val_ap` est dans `[0, 1]` et `val_mcc` est dans `[-1, 1]`

### Requirement: Les métriques déjà journalisées ne changent pas
La justesse, la précision, le rappel, le F1, les deux images de matrice de confusion et la perte SHALL garder leur définition et leur nom.

#### Scenario: Lot déjà connu
- **WHEN** les probabilités sont `0.9`, `0.8`, `0.2`, `0.1`, les cibles `1`, `0`, `1`, `0`, et le seuil `0.5`
- **THEN** la justesse, la précision, le rappel et le F1 valent chacun `0.5`
