import torch
from scipy import stats
from tasks import Task


def calc_variance(h_t: torch.Tensor) -> torch.Tensor:
    """Calculates variance for each input over trial and time.

    Parameters
    ----------
    h_t
        Activation of hidden state of model.
        Takes shape (T, batch_size (aka trial_size), hidden_size)

    Returns
    -------
    unit_var
        The variance for each unit over trial and time. Takes shape
        (hidden_size, )
    """
    # Puts all trials for a unit in the same coumn
    h_t = h_t.reshape(-1, h_t.shape[0])
    unit_var = h_t.var(axis=0)
    return unit_var


def task_variance(task: Task, model):

    # TODO: make this work
    thetas = torch.linspace(0, 2 * torch.pi, torch.pi / task.batch_size)
    periods = task.task_periods
    for stim_period in periods:
        if stim_period.name.contains("stimulus"):
            stim_period.add_stimulus_values(thetas)
    # Update the task z and thetas
    task.set_task_periods(periods)
    # TODO: clean this up
    h_t = model.evaluate(task)

    task_times = task.get_time_spans()[1:]
    period_variance = torch.Tensor((len(task.task_periods) - 1, model.hidden_size))
    for i, task_time_span in enumerate(task_times):
        h_t_slice = h_t[task_time_span["start"] : task_time_span["stop"]]
        per_var = calc_variance(h_t_slice)
        period_variance[i] = per_var

    return period_variance


def variance_correlation(task_1, task_2, model):
    task_variance_1 = task_variance(task_1, model)
    task_variance_2 = task_variance(task_2, model)
    # Need to sort and do correlation between rows
    corr = stats.pearsonr(task_variance_1, task_variance_2).statistic
    return corr
