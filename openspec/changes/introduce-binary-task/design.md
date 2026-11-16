# Design

## Context

Voir `proposal.md` (Why) et les specs `model-logits` et `binary-task`.
État observé :

- `Perceptron.forward` et `MLP.forward` terminent par `torch.sigmoid`.
  Le docstring du perceptron dit déjà « logit avant activation », puis
  le code applique la sigmoïde. Les deux modèles n'ont aucun paramètre
  dans cette sigmoïde : un checkpoint ne contient que les poids
  linéaires, sous le préfixe `model.`.
- `ClassificationModule` construit `nn.BCELoss()`, appelle `self(x)`
  et passe le résultat aux métriques avec `threshold` (`0.5` par
  défaut, groupe `module`). `forward` délègue au modèle.
- `Predictor.predict_proba` renvoie la sortie du modèle telle quelle.
  `predict` compare cette sortie au seuil. Le test d'intégration qui
  relit un checkpoint exige que cette probabilité égale `module(x)`.
- `run_test` construit aussi un `ClassificationModule`. La commande
  argparse n'a pas de groupe `task` : elle appelle le module avec le
  modèle, le taux et le seuil.
- `hydra_main` attrape `ValueError` avant le `Trainer`. Une
  `ValueError` levée dans un step sort dans la pile de Lightning, pas
  sur cette ligne d'erreur courte.
- `openspec/specs/` est vide. Les métriques (justesse, F1, AP, MCC)
  reçoivent déjà des probabilités : elles ne savent pas d'où elles
  viennent.

## Goals / Non-Goals

**Goals:**
- Une tâche est le seul endroit qui connaît la sigmoïde, la perte et
  la règle de décision. Le modèle s'arrête au logit.
- Pour les mêmes poids, la probabilité publiée reste la sigmoïde du
  logit, et la classe reste le seuil déjà configuré.
- Une cible incompatible échoue en Python, avec un message qui nomme
  l'intervalle ou les deux formes, avant l'appel de la perte.

**Non-Goals:**
- Enregistrer d'autres tâches ou d'autres pertes.
- Déplacer `threshold` ou `learning_rate`.
- Détecter à l'exécution qu'un modèle a déjà appliqué une sigmoïde.

## Decisions

### Le modèle s'arrête à la dernière couche linéaire

`Perceptron.forward` renvoie `self.linear(x)`. `MLP.forward` renvoie
`self.network(x)`, dont le dernier bloc est déjà `nn.Linear`. Les ReLU
cachées restent. Les comptes de paramètres des tests actuels (3, 17,
37) restent vrais, parce qu'une sigmoïde n'a pas de poids.

Alternative écartée : garder la sigmoïde dans le modèle et ne changer
que la perte. La tâche multiclasse voudrait un softmax, et une
entropie croisée sur des probabilités déjà saturées retombe sur le
problème numérique que cette change retire (R4 de l'audit).

### Une classe `BinaryTask`, construite par une factory

`ia_xp_plateform/task/binary.py` expose `BinaryTask`. Elle n'est pas
un `nn.Module`.

- `probabilities(logits)` renvoie `torch.sigmoid(logits)`.
- `loss(logits, targets)` vérifie la cible, puis appelle
  `nn.BCEWithLogitsLoss` (réduction moyenne, sans `pos_weight`).
- `predict(logits, threshold)` compare `probabilities(logits)` au
  seuil avec `>=`, puis renvoie un `long`.

`factory/task_factory.py` suit `transform_factory` : un dictionnaire
`TASKS` (`"binary"` → `BinaryTask`), `create_task(name, loss="bce")`.
Un nom absent lève `ValueError` en citant les noms connus, donc
`binary`. `BinaryTask` refuse une perte autre que `"bce"` en citant
`bce`. Les deux sont exportés depuis `factory/__init__.py`.

Le module fait `self.criterion = task.criterion` pour que le résumé
Lightning continue d'afficher la perte. `BCEWithLogitsLoss` sans
`pos_weight` n'a pas de paramètre : les clefs `model.*` d'un
checkpoint déjà écrit se rechargent.

Alternative écartée : un groupe `loss` séparé de l'activation. Deux
fichiers pourraient se contredire (`bce` avec un softmax) sans erreur.
La tâche les tient ensemble. Une perte future (`pos_weight`, focale)
sera un autre `loss` accepté par la même tâche, pas une nouvelle
tâche.

### Le module et le prédicteur publient la probabilité

`ClassificationModule.forward` renvoie
`task.probabilities(self.model(x))`. Le step, lui, appelle
`self.model(x)` pour obtenir les logits, puis `task.loss` et
`task.probabilities`. Les métriques reçoivent ces probabilités, comme
aujourd'hui. Ainsi `module(x)` et `predict_proba` restent égaux après
relecture d'un checkpoint.

`Predictor` prend une tâche, par défaut `create_task("binary")`.
`predict_proba` applique `task.probabilities`. `predict` appelle
`task.predict`. Le seuil reste l'argument du prédicteur et du module.

`ClassificationModule(..., task=None)` construit la tâche binaire.
`run_train` et `run_test` passent `create_task(args.task, loss=args.loss)`
quand ces attributs existent. L'argparse ne les a pas : le défaut du
module suffit, et aucun drapeau n'est ajouté.

Les modèles de test qui terminent par une `Sigmoid` la perdent, sinon
la tâche l'appliquerait une seconde fois.

### La vérification précède la perte

`loss` refuse d'abord les formes différentes, en nommant les deux
formes dans la `ValueError`. Elle refuse ensuite toute valeur
strictement inférieure à 0 ou strictement supérieure à 1, en nommant
l'intervalle `[0, 1]`. Une cible `0.3` passe : l'entropie croisée
binaire accepte une cible fractionnaire. La comparaison utilise le
tenseur déjà reçu, sans l'envoyer au critère.

Cette erreur naît dans le step. Lightning l'affiche dans sa pile ;
`hydra_main` ne la transforme pas en ligne unique. Le message reste
lisible dans cette pile. On ne déplace pas la vérification avant
`trainer.fit` : un lot de validation peut être le premier lot fautif.

Le scénario « tâche inconnue » est tenu par la factory, pas par le
texte d'erreur d'Hydra. Un fichier `conf/task/` absent échoue plus
tôt, à la composition.

### La configuration nomme la tâche, pas le seuil

`conf/task/binary.yaml` contient `name: binary` et `loss: bce`. La
ligne `- task: binary` est ajoutée aux `defaults` de `config.yaml`,
`train_perceptron.yaml` et `train_mlp.yaml`. Aucune autre clef de
`train_mlp.yaml` n'est modifiée.

`namespace_from_config` copie `cfg.task.name` et `cfg.task.loss`.
Le test de la configuration par défaut fige `binary` et `bce`. Les
configurations nommées vérifient seulement que le nom et la perte
sont des chaînes non vides, pour ne pas figer un choix d'expérience.

Alternative écartée : mettre `threshold` dans `conf/task/`. Les
surcharges `module.threshold` et l'argparse `--threshold` changeraient
de place sans gain pour cette seule tâche.

## Risks / Trade-offs

- [Un `BaseModel` qui sigmoïde encore est sigmoïdé deux fois] → le contrat et le README disent que `forward` renvoie le logit. Pas de détection automatique : une sortie dans `(0, 1)` est aussi un logit valide.
- [La courbe de perte n'est pas identique bit à bit à l'ancienne `BCELoss`] → pour les mêmes logits, `BCEWithLogitsLoss` égale l'entropie croisée de la sigmoïde, et reste finie quand la sigmoïde sature. Les probabilités, elles, restent la sigmoïde.
- [Le résumé Lightning affiche `BCEWithLogitsLoss` au lieu de `BCELoss`] → même rôle, aucun paramètre en plus. Les checkpoints déjà écrits se rechargent.
- [`train_mlp` avec 3 sorties et une cible à une colonne échoue toujours] → l'échec devient une `ValueError` qui nomme les formes `(batch, 3)` et `(batch, 1)`, à la place de l'assert CUDA. On ne corrige pas ce fichier d'expérience.
- [Une cible `NaN` passe le test d'intervalle] → les comparaisons `< 0` et `> 1` sont fausses pour `NaN`. Le spec ne couvre pas ce cas. La perte devient alors `NaN`, comme aujourd'hui avec une entrée non finie.

## Migration Plan

Aucune conversion de checkpoint. La sigmoïde n'était pas un poids :
`load_state_dict` sur `model.linear.*` ou `model.network.*` suffit.
`predict_proba` applique la sigmoïde, donc un checkpoint déjà écrit
redonne les mêmes probabilités.

Retour arrière : remettre la sigmoïde dans les deux `forward`,
`BCELoss` dans le module, et retirer le groupe `task` des `defaults`.
Les poids écrits entre-temps restent lisibles, puisqu'ils n'ont pas
changé de forme.
