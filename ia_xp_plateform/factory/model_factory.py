"""

La factory de modèles construit un `BaseModel` à partir de son nom. Le
reste de la plateforme (un futur entraîneur, une configuration) n'a pas
besoin d'importer `Perceptron` ni de connaître son constructeur : il donne
un nom et des paramètres, et récupère un modèle qui respecte le contrat
`forward`.

"""

from ia_xp_plateform.models.base import BaseModel
from ia_xp_plateform.models.perceptron import Perceptron

# Nom utilisé par la configuration -> classe concrète.
MODELS: dict[str, type[BaseModel]] = {
    "perceptron": Perceptron,
}


def create_model(name: str, **kwargs) -> BaseModel:
    """Construit le modèle enregistré sous `name`.

    Parameters
    ----------
    name : str
        Nom du modèle, une clé de `MODELS` (par exemple `"perceptron"`).
    **kwargs
        Arguments transmis tels quels au constructeur du modèle
        (`input_features`, `output_features`, ...).

    Returns
    -------
    BaseModel
        Une instance prête à être appelée avec `model(x)`.
    """
    try:
        model_cls = MODELS[name]
    except KeyError as error:
        known = ", ".join(sorted(MODELS))
        raise ValueError(
            f"Modèle inconnu : '{name}'. Modèles disponibles : {known}."
        ) from error
    return model_cls(**kwargs)
