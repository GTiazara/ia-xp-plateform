"""

La factory de tâches construit une tâche à partir de son nom. Le reste
de la plateforme n'importe pas `BinaryTask` ni `MulticlassTask` : il
donne un nom et, éventuellement, le nom de la perte.

"""

from ia_xp_plateform.task.binary import BinaryTask
from ia_xp_plateform.task.multiclass import MulticlassTask

# Nom utilisé par la configuration -> classe concrète.
TASKS: dict[str, type] = {
    "binary": BinaryTask,
    "multiclass": MulticlassTask,
}


def create_task(
    name: str,
    loss: str | None = None,
) -> BinaryTask | MulticlassTask:
    """Construit la tâche enregistrée sous `name`.

    Parameters
    ----------
    name : str
        Nom de la tâche, une clé de `TASKS`.
    loss : str | None, optional
        Nom de la perte. Absent : chaque tâche prend son propre défaut
        (`bce` ou `cross_entropy`).

    Returns
    -------
    BinaryTask | MulticlassTask
        Tâche prête à calculer une perte et une probabilité.

    Raises
    ------
    ValueError
        Si `name` n'est pas une tâche connue, ou si cette tâche refuse
        `loss`.
    """
    try:
        task_cls = TASKS[name]
    except KeyError as error:
        known = ", ".join(sorted(TASKS))
        raise ValueError(
            f"Tâche inconnue : '{name}'. Tâches disponibles : {known}."
        ) from error
    if loss is None:
        return task_cls()
    return task_cls(loss=loss)
