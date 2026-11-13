"""

Point d'entrée Hydra.

Le premier argument, s'il ne contient pas ``=``, est le nom d'un fichier
``conf/<nom>.yaml``. La suite sont des surcharges ``groupe.clef=valeur``.
Sans nom, on prend ``config``.

    poetry run ia-xp-conf
    poetry run ia-xp-conf train_mlp
    poetry run ia-xp-conf train_perceptron trainer.max_epochs=10

"""

import sys

from ia_xp_plateform.cli.from_config import launch
from ia_xp_plateform.configuration import available_configs


def parse_launch_arguments(argv: list[str]) -> tuple[str, list[str]]:
    """Sépare le nom de configuration et les surcharges.

    ``train_mlp trainer.max_epochs=5`` choisit ``conf/train_mlp.yaml``
    puis applique la surcharge. ``trainer.max_epochs=5`` seul garde
    ``config``.
    """
    if argv and "=" not in argv[0] and not argv[0].startswith("-"):
        return argv[0], list(argv[1:])
    return "config", list(argv)


def help_text() -> str:
    """Mode d'emploi, avec les fichiers présents dans ``conf/``."""
    names = "\n".join(f"  {name}" for name in available_configs())
    return f"""\
usage: ia-xp-conf [configuration] [groupe.clef=valeur ...]

configuration est un fichier conf/<nom>.yaml. Sans nom, on prend config.

configurations:
{names}

exemples:
  ia-xp-conf
  ia-xp-conf train_mlp
  ia-xp-conf train_perceptron trainer.max_epochs=10
  ia-xp-conf mode=test checkpoint=logs/xp01/checkpoints/best-model.ckpt
"""


def main(argv: list[str] | None = None) -> int:
    """Assemble la configuration et lance train, test ou predict."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments in (["-h"], ["--help"]):
        print(help_text(), end="")
        return 0
    config_name, overrides = parse_launch_arguments(arguments)
    try:
        launch(overrides, config_name=config_name)
    except (ValueError, FileNotFoundError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0
