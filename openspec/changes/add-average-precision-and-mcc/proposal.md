# Proposal

## Why

Sur un jeu à 88 % de négatifs, la justesse reste collée au taux de la
classe majoritaire, et la précision, le rappel et le F1 restent à 0
tant qu'aucune probabilité ne franchit 0,5. Deux modèles qui ne
prédisent jamais la classe 1 ont alors les mêmes courbes, même si l'un
range les vrais positifs en tête. Il manque une mesure de ce classement,
et une mesure qui utilise les quatre cases de la matrice.

## What Changes

- L'average precision (AP) d'une époque : aire sous la courbe
  précision-rappel, sans interpolation et sans seuil. Journalisée
  `train_ap`, `val_ap`, `test_ap`.
- Le coefficient de Matthews (MCC) de la même époque, calculé avec le
  seuil déjà utilisé par la précision et le rappel. Journalisé
  `train_mcc`, `val_mcc`, `test_mcc`.
- Les métriques déjà présentes (justesse, précision, rappel, F1,
  matrices de confusion, perte) ne changent pas.
- README : ce que l'AP et le MCC disent, et pourquoi ils restent
  informatifs quand le modèle répond toujours 0.

Hors périmètre : AUC ROC, score de Brier, courbe précision-rappel en
image, pondération de la perte, rééchantillonnage, seuil appris sur la
validation.

## Capabilities

### New Capabilities
- `binary-classification-metrics`: mesurer une époque de classification
  binaire par l'average precision et par le coefficient de Matthews, et
  les journaliser à côté des métriques à seuil déjà existantes.

### Modified Capabilities

Aucune : `openspec/specs/` est vide.

## Impact

- `ia_xp_plateform/metric/binary_classification.py` : l'AP demande de
  conserver les probabilités et les cibles de l'époque ; le MCC se
  calcule avec les compteurs déjà tenus.
- `ia_xp_plateform/module/classification.py` : les nouvelles clefs de
  `compute()` sont journalisées par la boucle existante. Les docstrings
  qui listent les noms de métriques citent `ap` et `mcc`.
- Tests unitaires de `BinaryClassificationMetrics` et test
  d'intégration du `fit` (présence de `val_ap` et `val_mcc`).
- README, section du module de classification.
- Aucune dépendance nouvelle. Le checkpoint, le seuil et la perte ne
  changent pas.
