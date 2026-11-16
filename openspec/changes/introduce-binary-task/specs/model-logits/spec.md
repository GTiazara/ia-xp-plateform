# Spec Delta

## Purpose

Les modèles tabulaires exposent le logit de leur dernière couche, pour qu'une tâche puisse choisir l'activation et la perte sans que le modèle les impose.

## ADDED Requirements

### Requirement: Le perceptron renvoie le logit
Le perceptron SHALL renvoyer la sortie de sa couche linéaire, sans la passer par une sigmoïde. La forme SHALL rester `(batch, output_features)`.

#### Scenario: Poids connus
- **WHEN** un perceptron à deux entrées a les poids `1` et `1`, le biais `0`, et reçoit la ligne `1`, `2`
- **THEN** la sortie vaut `3`

#### Scenario: Forme par défaut
- **WHEN** le perceptron par défaut reçoit un lot de 5 lignes à 2 colonnes
- **THEN** la sortie a la forme `(5, 1)`

### Requirement: Le MLP renvoie le logit de sa dernière couche
Le MLP SHALL renvoyer la sortie de sa dernière couche linéaire. Une valeur SHALL pouvoir sortir de `[0, 1]`. La forme SHALL rester `(batch, output_features)`.

#### Scenario: Logit hors de l'intervalle des probabilités
- **WHEN** la dernière couche d'un MLP produit une valeur strictement supérieure à 1
- **THEN** cette valeur est renvoyée telle quelle

#### Scenario: Plusieurs sorties
- **WHEN** un MLP a 4 entrées et 3 sorties et reçoit un lot de 2 lignes
- **THEN** la sortie a la forme `(2, 3)`

### Requirement: Le nombre de paramètres ne change pas
Retirer la sigmoïde SHALL laisser inchangé le nombre de paramètres.

#### Scenario: Perceptron minimal
- **WHEN** un perceptron a 2 entrées et 1 sortie
- **THEN** il a 3 paramètres

#### Scenario: MLP à une couche cachée
- **WHEN** un MLP a 2 entrées, une couche cachée de largeur 4 et 1 sortie
- **THEN** il a 17 paramètres

#### Scenario: MLP à deux couches cachées
- **WHEN** un MLP a 2 entrées, deux couches cachées de largeur 4 et 1 sortie
- **THEN** il a 37 paramètres

### Requirement: Un checkpoint de poids linéaires se recharge
Un dictionnaire de poids de couche linéaire SHALL se charger dans le perceptron, et l'appel du modèle SHALL redonner la combinaison linéaire de ces poids.

#### Scenario: Poids rechargés
- **WHEN** un perceptron à deux entrées charge les poids `1`, `1` et le biais `0`, puis reçoit la ligne `1`, `2`
- **THEN** la sortie vaut `3`
