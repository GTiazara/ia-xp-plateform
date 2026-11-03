"""ia-xp-plateform : plateforme d'expériences (XP) pour apprendre l'IA pas à pas.

Chaque "XP" est une expérience pédagogique autonome (un modèle, une tâche
d'apprentissage, un jeu de données) rangée dans ce package selon une
architecture modulaire inspirée de `bdf-model` :

    xp/
        train/
            models/   -> définition des modèles (ex: XP-01 Perceptron)
            module/    -> tâche d'apprentissage (forward + loss + métriques)
            data/      -> datasets et datamodules
        cli/           -> orchestration des expériences (Hydra)
        main.py        -> point d'entrée

Voir `docs/` pour la fiche de chaque XP.
"""

__version__ = "0.1.0"
