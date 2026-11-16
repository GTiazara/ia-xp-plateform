train-mlp:
	poetry run ia-xp-conf train_mlp

run-tensorboard:
	poetry run tensorboard --logdir logs/tensorboard