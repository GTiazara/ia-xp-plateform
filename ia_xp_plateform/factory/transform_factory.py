"""

La factory de transformations construit une transformation, ou les quatre
pipelines d'un prétraitement, à partir de leur nom. Le reste de la
plateforme n'importe pas `Standardize` ou `Clip` : il donne un nom et
des paramètres.

"""

from ia_xp_plateform.transform.elementwise import Clip, GaussianNoise
from ia_xp_plateform.transform.pipeline import Pipeline, Preprocessing
from ia_xp_plateform.transform.scaling import MinMax, Standardize
from ia_xp_plateform.transform.base import BaseTransform

# Nom utilisé par la configuration -> classe concrète.
TRANSFORMS: dict[str, type[BaseTransform]] = {
    "clip": Clip,
    "gaussian_noise": GaussianNoise,
    "min_max": MinMax,
    "standardize": Standardize,
}

_SPLITS = ("train", "val", "test", "predict")


def create_transform(name: str, **params) -> BaseTransform:
    """Construit la transformation enregistrée sous `name`."""
    try:
        transform_cls = TRANSFORMS[name]
    except KeyError as error:
        known = ", ".join(sorted(TRANSFORMS))
        raise ValueError(
            f"Transformation inconnue : '{name}'. "
            f"Transformations disponibles : {known}."
        ) from error
    try:
        return transform_cls(**params)
    except TypeError as error:
        raise ValueError(
            f"Paramètre refusé pour la transformation '{name}' : {error}"
        ) from error


def create_preprocessing(specs: dict | None) -> Preprocessing:
    """Construit les quatre pipelines décrits par `specs`.

    `None` et une clé de split absente donnent un pipeline vide. Chaque
    étape est un dictionnaire avec `name` et, éventuellement, les
    paramètres de la transformation.
    """
    specs = specs or {}
    pipelines = {
        split: Pipeline(_steps(specs.get(split) or []))
        for split in _SPLITS
    }
    return Preprocessing(**pipelines)


def _steps(steps: list) -> list[BaseTransform]:
    transforms = []
    for step in steps:
        if not isinstance(step, dict) or "name" not in step:
            raise ValueError("Chaque transformation doit avoir une clé 'name'.")
        params = {key: value for key, value in step.items() if key != "name"}
        transforms.append(create_transform(str(step["name"]), **params))
    return transforms
