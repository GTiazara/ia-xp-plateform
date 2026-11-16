# Spec Delta

## Purpose

Entraîner et mesurer une classification où chaque ligne a exactement une classe parmi les logits du modèle, et publier cette classe plutôt qu'une décision binaire par sortie.

## ADDED Requirements

### Requirement: La probabilité est le softmax des logits
Les probabilités SHALL être le softmax des logits. Elles SHALL être strictement positives et sommer à 1 sur chaque ligne.

#### Scenario: Logits égaux
- **WHEN** les logits d'une ligne sont `0`, `0`, `0`
- **THEN** chaque probabilité vaut `1/3`

### Requirement: La classe est l'indice du plus grand logit
La classe prédite SHALL être l'indice du plus grand logit. Deux logits égaux SHALL garder le plus petit indice. Le seuil du module SHALL ne pas changer cette classe.

#### Scenario: Un logit domine
- **WHEN** les logits sont `0`, `5`, `1`
- **THEN** la classe prédite vaut 1

#### Scenario: Égalité
- **WHEN** les logits sont `2`, `2`, `0`
- **THEN** la classe prédite vaut 0

#### Scenario: Le seuil est ignoré
- **WHEN** les mêmes logits sont décidés avec un seuil de `0.1` puis de `0.9`
- **THEN** les deux classes sont identiques

### Requirement: La perte est l'entropie croisée des logits
La perte SHALL être l'entropie croisée entre les logits et l'indice de classe. Elle SHALL rester finie lorsque un logit est très grand.

#### Scenario: Logits égaux et classe 0
- **WHEN** les logits sont `0`, `0`, `0` et la cible vaut `0`
- **THEN** la perte vaut `log(3)`

#### Scenario: Logit très grand sur une mauvaise classe
- **WHEN** les logits sont `80`, `0`, `0` et la cible vaut `1`
- **THEN** la perte est finie et strictement positive

### Requirement: La cible est un indice de classe
La tâche SHALL attendre au moins deux logits et une cible de forme `(N, 1)` ou `(N,)`. Chaque valeur SHALL être entière et dans `0 .. K-1`, où `K` est le nombre de logits. Une valeur `2.0` SHALL être l'indice 2. Sinon la tâche SHALL refuser le lot avant la perte, en nommant l'intervalle ou les formes.

#### Scenario: Indice entier accepté
- **WHEN** trois logits reçoivent la cible `2.0`
- **THEN** la perte est finie

#### Scenario: Indice hors intervalle
- **WHEN** trois logits reçoivent la cible `3`
- **THEN** l'erreur nomme l'intervalle allant de 0 à 2, et la perte n'est pas calculée

#### Scenario: Cible fractionnaire
- **WHEN** la cible vaut `1.5`
- **THEN** l'erreur est levée avant la perte

#### Scenario: Un seul logit
- **WHEN** le modèle ne renvoie qu'un logit
- **THEN** l'erreur nomme ce nombre de logits, et la perte n'est pas calculée

### Requirement: Les scores macro couvrent toutes les classes du modèle
La justesse SHALL être la part de classes justes. La précision, le rappel et le F1 SHALL être la moyenne non pondérée des scores de chacune des `K` classes, y compris une classe absente de l'époque. Une classe sans vrai positif et sans annonce SHALL avoir un F1 de 0.

#### Scenario: Une classe absente
- **WHEN** `K` vaut 3, les classes prédites sont `0`, `0`, `1` et les cibles `0`, `1`, `1`
- **THEN** la justesse vaut `2/3`, la précision macro vaut `1/2`, le rappel macro vaut `1/2` et le F1 macro vaut `4/9`

### Requirement: Le coefficient de Matthews multiclasse utilise toute la matrice
Le coefficient SHALL valoir `(c * s - somme_k p_k * t_k) / sqrt((s^2 - somme_k p_k^2) * (s^2 - somme_k t_k^2))`, où `c` est le nombre de prédictions justes, `s` le nombre d'exemples, `t_k` le nombre de cibles `k` et `p_k` le nombre de prédictions `k`. Un dénominateur nul SHALL donner 0.

#### Scenario: Accord parfait
- **WHEN** chaque classe du modèle est prédite exactement quand elle est la cible
- **THEN** le coefficient vaut 1

#### Scenario: Une seule classe des deux côtés
- **WHEN** toutes les cibles et toutes les prédictions valent la même classe
- **THEN** le coefficient vaut 0

### Requirement: La matrice a une case par couple de classes
La matrice de confusion SHALL être carrée, de côté `K`. La ligne SHALL être la classe réelle et la colonne la classe prédite. La matrice normalisée SHALL diviser chaque ligne par son total, et laisser à zéro une ligne vide.

#### Scenario: Trois exemples et une classe absente
- **WHEN** `K` vaut 3, les classes prédites sont `0`, `0`, `1` et les cibles `0`, `1`, `1`
- **THEN** la matrice d'effectifs est `[[1, 0, 0], [1, 1, 0], [0, 0, 0]]`

### Requirement: Les mesures multiclasses sont journalisées
Une époque de validation multiclasse SHALL publier `val_acc`, `val_precision`, `val_recall`, `val_f1` et `val_mcc`. `val_acc`, `val_precision`, `val_recall` et `val_f1` SHALL rester dans `[0, 1]`. `val_mcc` SHALL rester dans `[-1, 1]`. `val_ap` SHALL être absent.

#### Scenario: Fin d'époque
- **WHEN** une époque de validation de la tâche multiclasse se termine
- **THEN** `val_acc` et `val_f1` sont dans `[0, 1]`, `val_mcc` est dans `[-1, 1]`, et `val_ap` n'est pas publié

### Requirement: La configuration nomme la tâche multiclasse
Le fichier de tâche `multiclass` SHALL choisir la perte `cross_entropy`. Une perte inconnue pour cette tâche SHALL être refusée par une erreur qui cite `cross_entropy`. Une tâche inconnue SHALL être refusée par une erreur qui cite `binary` et `multiclass`. La configuration par défaut SHALL rester la tâche `binary`.

#### Scenario: Fichier multiclass
- **WHEN** la tâche `multiclass` est assemblée
- **THEN** son nom est `multiclass` et sa perte est `cross_entropy`

#### Scenario: Perte inconnue
- **WHEN** la tâche multiclasse reçoit une perte autre que `cross_entropy`
- **THEN** l'erreur cite `cross_entropy`

#### Scenario: Tâche inconnue
- **WHEN** on demande une tâche dont le nom n'est pas enregistré
- **THEN** l'erreur cite `binary` et `multiclass`

### Requirement: Le fichier de prédiction annonce une seule classe
Pour la tâche multiclasse, le fichier écrit SHALL contenir `id`, `prediction` et `probability_0` … `probability_{K-1}`. `prediction` SHALL être l'indice de classe. Les probabilités d'une ligne SHALL sommer à 1.

#### Scenario: Trois classes
- **WHEN** la prédiction multiclasse porte sur trois logits
- **THEN** le fichier contient `id`, `prediction`, `probability_0`, `probability_1` et `probability_2`

### Requirement: La prédiction binaire à une sortie ne change pas
Quand la tâche est binaire et que le modèle a une sortie, le fichier SHALL garder les colonnes `id`, `probability` et `prediction`.

#### Scenario: Tâche binaire
- **WHEN** la prédiction binaire porte sur une seule sortie
- **THEN** le fichier contient `id`, `probability` et `prediction`
