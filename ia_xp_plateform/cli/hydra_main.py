"""

Point d'entrée Hydra.

Les arguments ne sont pas des options ``--data``. Ce sont des surcharges
de la configuration, au format ``groupe.clef=valeur`` :

    poetry run ia-xp-conf data.data_path=data.csv trainer.max_epochs=10
    poetry run ia-xp-conf mode=test checkpoint=logs/xp01/checkpoints/best-model.ckpt
    poetry run ia-xp-conf mode=predict checkpoint=logs/xp01/checkpoints/best-model.ckpt predictions=out.csv

"""

import sys

from ia_xp_plateform.cli.from_config import launch

_HELP = """\
usage: ia-xp-conf [groupe.clef=valeur ...]

Assemble conf/config.yaml puis lance le mode choisi (train par défaut).

exemples:
  ia-xp-conf data.data_path=data.csv trainer.max_epochs=10
  ia-xp-conf mode=test checkpoint=logs/xp01/checkpoints/best-model.ckpt
  ia-xp-conf mode=predict checkpoint=logs/xp01/checkpoints/best-model.ckpt predictions=out.csv
"""


def main(argv: list[str] | None = None) -> int:
    """Assemble la configuration et lance train, test ou predict."""
    overrides = list(sys.argv[1:] if argv is None else argv)
    if overrides in (["-h"], ["--help"]):
        print(_HELP, end="")
        return 0
    try:
        launch(overrides)
    except (ValueError, FileNotFoundError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0
