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


class _OfflineSeries:
    """Remplace une série LitLogger : on retient les valeurs, sans réseau."""

    def __init__(self) -> None:
        self.values: list[float] = []

    def append(self, value, step=None) -> None:
        self.values.append(value)


class OfflineExperiment:
    """Expérience LitLogger locale.

    La vraie classe ouvre une connexion lightning.ai dès sa construction.
    Les tests entraînent sans compte : ils vérifient que les métriques
    arrivent bien au logger, pas qu'elles sont publiées.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.url = "offline://litlogger"
        self.series: dict[str, _OfflineSeries] = {}

    def __getitem__(self, key: str) -> _OfflineSeries:
        return self.series.setdefault(key, _OfflineSeries())

    def __setitem__(self, key: str, value) -> None:
        return None

    def finalize(self, status=None) -> None:
        return None

    def print_url(self) -> None:
        return None


@pytest.fixture(autouse=True)
def offline_litlogger(monkeypatch):
    """Empêche un entraînement de test d'ouvrir le navigateur de connexion."""
    import litlogger

    monkeypatch.setattr(litlogger, "Experiment", OfflineExperiment)
