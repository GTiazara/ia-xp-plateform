# Spec Delta

## Purpose

La tâche binaire transforme les logits d'un modèle en probabilité, en perte et en classe, et elle refuse une cible que cette perte ne sait pas mesurer.

## ADDED Requirements

### Requirement: La probabilité est la sigmoïde du logit
L'appel du module d'entraînement et la probabilité prédite SHALL valoir la sigmoïde des logits du modèle, donc une valeur dans `(0, 1)`.

#### Scenario: Logit nul
- **WHEN** le modèle renvoie le logit `0`
- **THEN** la probabilité vaut `1/2`

#### Scenario: Même probabilité des deux côtés
- **WHEN** un checkpoint de perceptron est relu par le prédicteur et par le module d'entraînement
- **THEN** les deux probabilités sont égales

### Requirement: La classe se décide sur la probabilité
La classe prédite SHALL valoir 1 lorsque la probabilité est supérieure ou égale au seuil, et 0 sinon. Le seuil SHALL être celui déjà configuré pour le module, `0.5` par défaut. La comparaison SHALL porter sur la probabilité, pas sur le logit.

#### Scenario: Seuil par défaut
- **WHEN** le logit vaut `1` et le seuil vaut `0.5`
- **THEN** la classe prédite vaut 1

#### Scenario: Seuil au-dessus de la probabilité
- **WHEN** le logit vaut `1` et le seuil vaut `0.9`
- **THEN** la classe prédite vaut 0

### Requirement: La perte est l'entropie croisée binaire des logits
La perte SHALL être l'entropie croisée binaire calculée sur les logits. Elle SHALL rester finie lorsque la sigmoïde du logit vaut 0 ou 1 à la précision de la machine.

#### Scenario: Logit nul et cible positive
- **WHEN** le logit vaut `0` et la cible vaut `1`
- **THEN** la perte vaut `log(2)`

#### Scenario: Logit très grand et cible négative
- **WHEN** le logit vaut `80` et la cible vaut `0`
- **THEN** la perte est finie et strictement positive

### Requirement: Une cible hors de l'intervalle est refusée
La tâche binaire SHALL refuser, avant de calculer la perte, une cible dont une valeur est hors de `[0, 1]`. L'erreur SHALL nommer cet intervalle.

#### Scenario: Classe entière hors du binaire
- **WHEN** une cible vaut `2`
- **THEN** le calcul lève une erreur qui nomme l'intervalle `[0, 1]`, et la perte n'est pas calculée

#### Scenario: Cible fractionnaire admise
- **WHEN** la cible vaut `0.3` et a la même forme que les logits
- **THEN** la perte est finie

### Requirement: Une forme différente est refusée
La tâche binaire SHALL refuser, avant de calculer la perte, des logits et une cible de formes différentes. L'erreur SHALL nommer les deux formes.

#### Scenario: Trois sorties pour une cible
- **WHEN** les logits ont la forme `(2, 3)` et la cible la forme `(2, 1)`
- **THEN** le calcul lève une erreur qui nomme ces deux formes, et la perte n'est pas calculée

### Requirement: La configuration par défaut choisit la tâche binaire
La configuration assemblée par défaut SHALL choisir la tâche de nom `binary` et la perte de nom `bce`. Une tâche inconnue SHALL être refusée par une erreur qui cite `binary`. Une perte inconnue pour cette tâche SHALL être refusée par une erreur qui cite `bce`.

#### Scenario: Configuration par défaut
- **WHEN** la configuration par défaut est assemblée
- **THEN** la tâche s'appelle `binary` et sa perte s'appelle `bce`

#### Scenario: Tâche inconnue
- **WHEN** on demande une tâche dont le nom n'est pas enregistré
- **THEN** l'erreur cite `binary`

#### Scenario: Perte inconnue
- **WHEN** la tâche binaire reçoit une perte autre que `bce`
- **THEN** l'erreur cite `bce`

### Requirement: Le fichier de prédiction binaire ne change pas
Pour une seule sortie, le fichier écrit par la prédiction SHALL garder les colonnes `id`, `probability` et `prediction`. `probability` SHALL être dans `(0, 1)` et `prediction` SHALL valoir 0 ou 1.

#### Scenario: Une sortie
- **WHEN** la prédiction porte sur un modèle à une sortie
- **THEN** le fichier contient les colonnes `id`, `probability` et `prediction`

### Requirement: Omettre la tâche laisse l'entraînement binaire
Construire le module d'entraînement sans nommer de tâche SHALL utiliser la tâche binaire.

#### Scenario: Module construit sans tâche nommée
- **WHEN** un pas d'entraînement reçoit des logits et des cibles `0` ou `1` de même forme, sans tâche nommée
- **THEN** la perte renvoyée est finie et strictement positive
