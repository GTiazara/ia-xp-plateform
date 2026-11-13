"""Dessine la matrice de confusion en image.

TensorBoard sait afficher une figure. Le calcul des comptes reste dans
`BinaryClassificationMetrics` : ce module normalise ces comptes, puis
dessine soit les effectifs, soit la part de chaque classe réelle.
"""

from io import BytesIO

import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def normalize_confusion_matrix(matrix) -> torch.Tensor:
    """Part de chaque prédiction dans sa classe réelle.

    Chaque ligne est divisée par son total, donc les deux cases d'une
    ligne non vide somment à 1. Une ligne vide (aucun exemple de cette
    classe) reste à zéro.

    Parameters
    ----------
    matrix
        Tenseur `[[vrais négatifs, faux positifs], [faux négatifs, vrais positifs]]`.

    Returns
    -------
    torch.Tensor
        Même forme, avec des proportions entre 0 et 1.
    """
    values = matrix.detach().float()
    row_sums = values.sum(dim=1, keepdim=True)
    return values / row_sums.clamp(min=1.0)


def confusion_matrix_figure(matrix, title: str, *, normalized: bool = False) -> Figure:
    """Construit l'image d'une matrice 2×2.

    Parameters
    ----------
    matrix
        Tenseur `[[vrais négatifs, faux positifs], [faux négatifs, vrais positifs]]`,
        en effectifs ou déjà normalisé.
    title : str
        Titre de la figure, par exemple « validation ».
    normalized : bool, optional
        `True` affiche des proportions entre 0 et 1, avec une échelle
        de couleur fixe. Par défaut `False` : les effectifs entiers.

    Returns
    -------
    matplotlib.figure.Figure
        Figure prête à être envoyée à un logger (`add_figure`).
    """
    values = matrix.detach().cpu().numpy()
    figure = Figure(figsize=(4, 4))
    FigureCanvasAgg(figure)
    ax = figure.add_subplot(111)
    if normalized:
        image = ax.imshow(values, cmap="Blues", vmin=0.0, vmax=1.0)
    else:
        image = ax.imshow(values, cmap="Blues")
    figure.colorbar(image, ax=ax)
    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_title(title)

    darkest = values.max() if values.size else 0
    for row in range(2):
        for column in range(2):
            value = values[row, column]
            color = "white" if darkest and value > darkest / 2 else "black"
            label = f"{value:.2f}" if normalized else f"{int(value)}"
            ax.text(column, row, label, ha="center", va="center", color=color)
    figure.tight_layout()
    return figure


def confusion_matrix_png(matrix, title: str, *, normalized: bool = False) -> bytes:
    """Même figure, encodée en PNG."""
    figure = confusion_matrix_figure(matrix, title, normalized=normalized)
    buffer = BytesIO()
    figure.savefig(buffer, format="png")
    return buffer.getvalue()
