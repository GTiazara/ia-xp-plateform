"""Tests unitaires de `BinaryClassificationMetrics`.

Les chiffres sont vérifiés sur un lot dont on connaît le résultat,
sans module d'entraînement ni Trainer.
"""

import pytest
import torch

from ia_xp_plateform.metric import BinaryClassificationMetrics


def test_precision_recall_and_accuracy_on_a_known_batch():
    metrics = BinaryClassificationMetrics()
    # Un vrai positif, un faux positif, un faux négatif, un vrai négatif.
    probabilities = torch.tensor([[0.9], [0.8], [0.2], [0.1]])
    targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

    metrics.update(probabilities, targets, threshold=0.5)
    values = metrics.compute()

    assert values["acc"].item() == pytest.approx(0.5)
    assert values["precision"].item() == pytest.approx(0.5)
    assert values["recall"].item() == pytest.approx(0.5)
    assert values["f1"].item() == pytest.approx(0.5)


def test_precision_and_recall_are_zero_when_nothing_is_predicted_positive():
    metrics = BinaryClassificationMetrics()
    probabilities = torch.zeros(2, 1)
    targets = torch.tensor([[1.0], [0.0]])

    metrics.update(probabilities, targets, threshold=0.5)
    values = metrics.compute()

    assert values["precision"].item() == 0.0
    assert values["recall"].item() == 0.0
    assert values["f1"].item() == 0.0
    assert values["acc"].item() == pytest.approx(0.5)


def test_two_batches_are_counted_together_until_reset():
    metrics = BinaryClassificationMetrics()
    metrics.update(torch.tensor([[0.9]]), torch.tensor([[1.0]]), threshold=0.5)
    metrics.update(torch.tensor([[0.1]]), torch.tensor([[0.0]]), threshold=0.5)

    assert metrics.compute()["acc"].item() == pytest.approx(1.0)

    metrics.reset()

    assert metrics.compute()["acc"].item() == 0.0
    assert metrics.compute()["precision"].item() == 0.0
    assert metrics.compute()["recall"].item() == 0.0
    assert metrics.compute()["f1"].item() == 0.0


def test_f1_is_the_harmonic_mean_of_precision_and_recall():
    metrics = BinaryClassificationMetrics()
    # Un vrai positif, deux faux positifs : précision 1/3, rappel 1, F1 0,5.
    probabilities = torch.tensor([[0.9], [0.9], [0.9]])
    targets = torch.tensor([[1.0], [0.0], [0.0]])

    metrics.update(probabilities, targets, threshold=0.5)
    values = metrics.compute()

    assert values["precision"].item() == pytest.approx(1 / 3)
    assert values["recall"].item() == pytest.approx(1.0)
    assert values["f1"].item() == pytest.approx(0.5)


def test_average_precision_is_one_when_the_ranking_is_perfect_below_the_threshold():
    metrics = BinaryClassificationMetrics()
    metrics.update(
        torch.tensor([0.4, 0.3, 0.2, 0.1]),
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
        threshold=0.5,
    )
    values = metrics.compute()

    assert values["ap"].item() == pytest.approx(1.0)
    assert values["precision"].item() == 0.0
    assert values["recall"].item() == 0.0
    assert values["f1"].item() == 0.0


def test_average_precision_of_an_imperfect_ranking_is_five_sixths():
    metrics = BinaryClassificationMetrics()
    metrics.update(
        torch.tensor([0.9, 0.8, 0.1]),
        torch.tensor([1.0, 0.0, 1.0]),
        threshold=0.5,
    )

    assert metrics.compute()["ap"].item() == pytest.approx(5 / 6)


def test_equal_scores_keep_the_arrival_order():
    metrics = BinaryClassificationMetrics()
    metrics.update(
        torch.tensor([0.5, 0.5]),
        torch.tensor([0.0, 1.0]),
        threshold=0.5,
    )

    assert metrics.compute()["ap"].item() == pytest.approx(0.5)


def test_average_precision_is_zero_without_a_positive_or_an_example():
    metrics = BinaryClassificationMetrics()

    assert metrics.compute()["ap"].item() == 0.0

    metrics.update(torch.tensor([0.2, 0.9]), torch.tensor([0.0, 0.0]), threshold=0.5)

    assert metrics.compute()["ap"].item() == 0.0


def test_average_precision_accumulates_batches_until_reset():
    metrics = BinaryClassificationMetrics()
    metrics.update(torch.tensor([0.9]), torch.tensor([1.0]), threshold=0.5)
    metrics.update(torch.tensor([0.1]), torch.tensor([0.0]), threshold=0.5)

    assert metrics.compute()["ap"].item() == pytest.approx(1.0)

    metrics.reset()

    assert metrics.compute()["ap"].item() == 0.0


def test_matthews_correlation_is_one_when_every_prediction_is_right():
    metrics = BinaryClassificationMetrics()
    metrics.update(
        torch.tensor([0.9, 0.8, 0.2, 0.1]),
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
        threshold=0.5,
    )

    assert metrics.compute()["mcc"].item() == pytest.approx(1.0)


def test_matthews_correlation_is_minus_one_when_every_prediction_is_wrong():
    metrics = BinaryClassificationMetrics()
    metrics.update(
        torch.tensor([0.9, 0.8, 0.2, 0.1]),
        torch.tensor([0.0, 0.0, 1.0, 1.0]),
        threshold=0.5,
    )

    assert metrics.compute()["mcc"].item() == pytest.approx(-1.0)


def test_matthews_correlation_of_a_mixed_matrix_is_five_twelfths():
    metrics = BinaryClassificationMetrics()
    metrics.update(
        torch.tensor([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2]),
        torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]),
        threshold=0.5,
    )

    assert metrics.compute()["mcc"].item() == pytest.approx(5 / 12)


def test_matthews_correlation_is_zero_when_every_prediction_is_negative():
    metrics = BinaryClassificationMetrics()
    metrics.update(
        torch.tensor([0.2, 0.1]),
        torch.tensor([1.0, 0.0]),
        threshold=0.5,
    )

    assert metrics.compute()["mcc"].item() == 0.0


def test_two_outputs_of_one_example_count_as_two_binary_examples():
    metrics = BinaryClassificationMetrics()
    metrics.update(
        torch.tensor([[0.9, 0.1]]),
        torch.tensor([[1.0, 0.0]]),
        threshold=0.5,
    )

    assert metrics.compute()["ap"].item() == pytest.approx(1.0)


def test_known_batch_keeps_accuracy_precision_recall_and_f1():
    metrics = BinaryClassificationMetrics()
    metrics.update(
        torch.tensor([0.9, 0.8, 0.2, 0.1]),
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
        threshold=0.5,
    )
    values = metrics.compute()

    assert values["acc"].item() == pytest.approx(0.5)
    assert values["precision"].item() == pytest.approx(0.5)
    assert values["recall"].item() == pytest.approx(0.5)
    assert values["f1"].item() == pytest.approx(0.5)


def test_confusion_matrix_counts_true_and_predicted_classes():
    metrics = BinaryClassificationMetrics()
    probabilities = torch.tensor([[0.9], [0.8], [0.2], [0.1]])
    targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

    metrics.update(probabilities, targets, threshold=0.5)

    assert metrics.confusion_matrix().tolist() == [[1.0, 1.0], [1.0, 1.0]]


def test_confusion_matrix_figure_is_a_png():
    from ia_xp_plateform.metric import confusion_matrix_png

    png = confusion_matrix_png(torch.tensor([[3.0, 1.0], [2.0, 4.0]]), "validation")

    assert png.startswith(b"\x89PNG")


def test_normalized_confusion_matrix_is_the_share_of_each_true_class():
    from ia_xp_plateform.metric import normalize_confusion_matrix

    matrix = torch.tensor([[90.0, 10.0], [80.0, 20.0]])
    normalized = normalize_confusion_matrix(matrix)

    assert normalized[0].tolist() == pytest.approx([0.9, 0.1])
    assert normalized[1].tolist() == pytest.approx([0.8, 0.2])


def test_normalized_confusion_matrix_keeps_an_empty_class_at_zero():
    from ia_xp_plateform.metric import confusion_matrix_png, normalize_confusion_matrix

    matrix = torch.tensor([[0.0, 0.0], [0.0, 4.0]])
    normalized = normalize_confusion_matrix(matrix)

    assert normalized[0].tolist() == pytest.approx([0.0, 0.0])
    assert normalized[1].tolist() == pytest.approx([0.0, 1.0])
    assert confusion_matrix_png(normalized, "validation", normalized=True).startswith(b"\x89PNG")
