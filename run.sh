#!/usr/bin/env bash
# Exemple de lancement de l'XP-01 (perceptron / voiture virtuelle).
#
# Usage:
#   ./run.sh                 # lance l'entraînement avec la config par défaut
#   ./run.sh trainer.max_epochs=5 datamodule.batch_size=64

poetry run python xp/main.py "$@"
