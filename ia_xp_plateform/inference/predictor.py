"""

L'inférence consiste à utiliser un modèle déjà entraîné pour prédire, sans
rien relancer de l'entraînement : pas de `Trainer`, pas de datamodule, pas
de perte ni d'optimiseur. Il suffit de deux choses : l'architecture
(un `BaseModel`) et ses poids (le checkpoint écrit pendant le fit).

`Predictor` réunit les deux. Il met le modèle en mode évaluation, coupe le
calcul des gradients, et demande à la tâche binaire la probabilité et la
classe, avec le même seuil que `ClassificationModule`.

"""

from pathlib import Path

import torch

from ia_xp_plateform.factory.model_factory import create_model
from ia_xp_plateform.factory.task_factory import create_task
from ia_xp_plateform.models.base import BaseModel
from ia_xp_plateform.task.binary import BinaryTask

# Dans le checkpoint de `ClassificationModule`, les poids du modèle sont
# rangés sous le nom de l'attribut `self.model` : « model.linear.weight »
# au lieu de « linear.weight ». On retire ce préfixe pour retrouver les
# noms que connaît le `BaseModel` seul.
_MODEL_PREFIX = "model."


class Predictor:
    """Prédit avec un `BaseModel` entraîné, indépendamment du fit.

    Deux façons de le construire :

    1. À partir d'un modèle déjà prêt : `Predictor(model)`.
    2. À partir d'un checkpoint d'entraînement :
       `Predictor.from_checkpoint("best-model.ckpt", "perceptron")`.
    """

    def __init__(
        self,
        model: BaseModel,
        threshold: float = 0.5,
        device: str | torch.device = "cpu",
        task: BinaryTask | None = None,
    ) -> None:
        """Prépare le modèle pour l'inférence.

        Le modèle reçu est déplacé sur `device` et passé en mode
        évaluation (`model.eval()`), c'est-à-dire modifié sur place.

        Parameters
        ----------
        model : BaseModel
            Modèle dont les poids sont déjà ceux de l'entraînement. Il doit
            renvoyer un logit. La tâche en fait une probabilité.
        threshold : float, optional
            Seuil au-dessus duquel la probabilité devient la classe `1`.
            Par défaut `0.5`, comme `ClassificationModule`.
        device : str | torch.device, optional
            Appareil de calcul. Par défaut `"cpu"`.
        task : BinaryTask | None, optional
            Tâche qui calcule la probabilité et la classe. Par défaut, la
            tâche binaire.
        """
        if not isinstance(model, BaseModel):
            raise TypeError(
                "model doit hériter de BaseModel "
                f"(reçu : {type(model).__name__})."
            )
        if not 0.0 < threshold < 1.0:
            raise ValueError(
                f"threshold doit être entre 0 et 1 ({threshold})."
            )

        self.device = torch.device(device)
        self.threshold = threshold
        self.task = create_task("binary") if task is None else task
        self.model = model.to(self.device)
        # En évaluation, les couches comme Dropout ou BatchNorm cessent de
        # se comporter comme pendant l'entraînement : sans cela, deux appels
        # sur la même entrée pourraient donner deux résultats différents.
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_name: str,
        model_kwargs: dict | None = None,
        threshold: float = 0.5,
        device: str | torch.device = "cpu",
        task: BinaryTask | None = None,
    ) -> "Predictor":
        """Reconstruit le modèle et charge ses poids depuis un checkpoint.

        Le checkpoint ne contient que les poids : l'architecture doit être
        redonnée, avec les mêmes paramètres que pendant l'entraînement.

        Parameters
        ----------
        checkpoint_path : str | Path
            Fichier `.ckpt` écrit par `ModelCheckpoint`
            (par exemple `output_dir/checkpoints/best-model.ckpt`).
        model_name : str
            Nom du modèle dans la factory (par exemple `"perceptron"`).
        model_kwargs : dict | None, optional
            Paramètres du constructeur du modèle
            (par exemple `{"input_features": 2}`).
        threshold : float, optional
            Seuil de classification. Par défaut `0.5`.
        device : str | torch.device, optional
            Appareil de calcul. Par défaut `"cpu"`.
        task : BinaryTask | None, optional
            Tâche qui calcule la probabilité. Par défaut, la tâche binaire.

        Returns
        -------
        Predictor
            Prêt à prédire.

        Raises
        ------
        FileNotFoundError
            Si `checkpoint_path` n'existe pas.
        ValueError
            Si le fichier n'est pas un checkpoint de `ClassificationModule`,
            ou si ses poids ne correspondent pas au modèle demandé.
        """
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint introuvable : {path}")

        # `weights_only=True` refuse d'exécuter du code arbitraire contenu
        # dans le fichier (un checkpoint est un fichier pickle).
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError(
                f"{path} n'est pas un checkpoint Lightning "
                "(clé 'state_dict' absente)."
            )

        weights = {
            key[len(_MODEL_PREFIX):]: value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith(_MODEL_PREFIX)
        }
        if not weights:
            raise ValueError(
                f"{path} ne contient aucun poids sous le préfixe "
                f"'{_MODEL_PREFIX}' : ce n'est pas un checkpoint de "
                "ClassificationModule."
            )

        model = create_model(model_name, **(model_kwargs or {}))
        try:
            model.load_state_dict(weights)
        except RuntimeError as error:
            raise ValueError(
                f"Les poids de {path} ne correspondent pas au modèle "
                f"'{model_name}' avec les paramètres {model_kwargs or {}}."
            ) from error

        return cls(model, threshold=threshold, device=device, task=task)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        """Logits du modèle, sans gradient."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"x doit être un torch.Tensor (reçu : {type(x).__name__})."
            )
        # Pas de gradient : on ne met aucun poids à jour, et le calcul est
        # plus léger en mémoire.
        with torch.no_grad():
            return self.model(x.to(self.device))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Probabilité de la classe `1` pour chaque exemple.

        Parameters
        ----------
        x : torch.Tensor
            Entrées, de forme `(batch, n_features)`.

        Returns
        -------
        torch.Tensor
            Sigmoïde des logits, entre 0 et 1, sur CPU, sans gradient.
        """
        return self.task.probabilities(self._logits(x)).cpu()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Classe prédite (`0` ou `1`) pour chaque exemple.

        Parameters
        ----------
        x : torch.Tensor
            Entrées, de forme `(batch, n_features)`.

        Returns
        -------
        torch.Tensor
            Entiers `0` ou `1`, de même forme que la sortie du modèle.
            Le seuil s'applique à la probabilité, pas au logit.
        """
        return self.task.predict(self._logits(x), self.threshold).cpu()
