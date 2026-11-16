# Spec Delta

## Purpose

Rendre un entraînement sur CPU reproductible à partir de la graine de l'expérience, sans déplacer les lignes du découpage.

## ADDED Requirements

### Requirement: La même graine reproduit les poids sur CPU
La commande d'entraînement SHALL semer l'initialisation et l'ordre des batches avec la graine de l'expérience. Deux entraînements sur CPU SHALL produire les mêmes poids lorsqu'ils partagent cette graine, les données, l'architecture et le nombre d'époques.

#### Scenario: Deux entraînements avec la graine 7
- **WHEN** deux commandes d'entraînement sur CPU utilisent la graine `7`, le même tableau et une époque
- **THEN** les poids des deux checkpoints sont identiques

#### Scenario: Une autre graine change les poids
- **WHEN** un troisième entraînement sur CPU, identique, utilise la graine `8`
- **THEN** au moins un poids diffère de ceux obtenus avec la graine `7`

### Requirement: Le découpage reste fixé par sa graine
Semer l'entraînement SHALL laisser les mêmes lignes dans chaque part lorsque la graine de découpage ne change pas.

#### Scenario: Le générateur global change
- **WHEN** le même tableau est découpé deux fois avec la même graine de découpage, après deux semis globaux différents
- **THEN** les identifiants de validation sont les mêmes

### Requirement: L'aide de l'entraînement décrit cette graine
L'aide de la commande d'entraînement SHALL dire que sa graine sème l'initialisation et l'ordre des batches, en plus du découpage.

#### Scenario: Aide de train
- **WHEN** l'utilisateur affiche l'aide de la commande d'entraînement
- **THEN** le texte de la graine contient « initialisation »
