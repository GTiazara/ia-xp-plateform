"""Fixtures partagées par les tests d'intégration."""

import pandas as pd
import pytest


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Petit tableau tabulaire : un `id`, deux attributs, une cible.

    20 lignes, assez pour vérifier un découpage 70 / 20 / 10 % sans
    arrondi ambigu (14 / 4 / 2 lignes).
    """
    return pd.DataFrame(
        {
            "id": range(20),
            "a": [float(i) for i in range(20)],
            "b": [float(i) * 2 for i in range(20)],
            "target": [float(i % 2) for i in range(20)],
        }
    )
