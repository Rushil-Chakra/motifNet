from typing import Union, Optional
from tasks import Task

from network import MotifNetwork
from train import train
from tasks import TASK_VECTOR_DEFINITION
import task_init

import torch
import matplotlib.pyplot as plt

import pickle
import logging
import argparse

from FixedPointFinder.FixedPointFinderTorch import (
    FixedPointFinderTorch as FixedPointFinder,
)
from FixedPointFinder.plot_utils import plot_fps

logger = logging.getLogger(__name__)

SEED = 42
rng = torch.manual_seed(SEED)


def run_training(
    exp_params: Optional[dict] = None,
    task_list: Union[Task, list[Task]] = None,
) -> torch.nn:
    """Train a network on a a set of tasks.

    Parameters
    ----------
    exp_params
        Parameters to initialize experiment with.
    task_list
        list of tasks to train network on. Must be either a list of
        tasks or a single task.

    Returns
    -------
    model
        The trained network. This is also saved into a model directory
    """
    if isinstance(task_list, Task):
        task_list = [task_list]
    model_kwargs = {"seed": SEED, "nonlinearity": "softplus", "g": 0.8}
    model = MotifNetwork(**model_kwargs)

    if exp_params is None:
        exp_params = {
            "lr": 1e-3,
            "l2_terms": (1e-6, 1e-6),
            "model": model,
            "batch_size": 64,
            "max_iter": 50000000,
            "clip_grad": True,
            "max_norm": 10,
            "task_list": task_list,
            "task_kwargs": {"gamma": 0.2, "rng": rng},
        }

    model = train(**exp_params)
    logger.info("saving model")
    # TODO: Save model with some parameter set as name
    torch.save(model, "trained_model.pt")
    return model


def find_fixed_points(trained_model, batch_size):
    # Get the trained model's hidden weights and input
    task = task_init.MemoryPro(batch_size=batch_size)
    u = task.get_input_array()
    h_t = trained_model.init_hidden(batch_size)
    with torch.no_grad():
        output, h_t = trained_model(u, h_t)

    # TODO: Define why im doing this, I know it has to do with slicing weights on time period
    # to get separate dynamics for each trial period
    def find_fixed_points_period(h_t, period_times):
        h_t_slice = (
            h_t[period_times["start"] : period_times["end"]].cpu().detach().numpy()
        )
        return h_t_slice

    task_times = task.get_time_spans()
    input_state_periods = [
        period.input_array.detach().cpu().numpy() for period in task.task_periods
    ]
    hidden_state_periods = [
        find_fixed_points_period(h_t, period_times)
        for period_times in task_times.values()
    ]
    fpf = FixedPointFinder(
        trained_model,
        method="joint",
        tol_q=1e-9,
        max_iters=10000,
        outlier_distance_scale=50.0,
    )

    for period_input, period_hidden_state, period in zip(
        input_state_periods, hidden_state_periods, task.task_periods
    ):
        inputs, state_trajectory = fpf.sample_inputs_and_states(
            period_input, period_hidden_state, n_inits=1000, noise_scale=0.05
        )
        unique_fps, all_fps = fpf.find_fixed_points(state_trajectory, inputs)
        fig = plot_fps(unique_fps, period_hidden_state, title=period)
        with open(f"./figures/{period}.pickle", "wb"):
            pickle.dump(fig)

        # inputs, state_traj = fpf.sample_inputs_and_states(
        #     u.detach().cpu().numpy(),
        #     h_t.detach().cpu().numpy(),
        #     n_inits=4000,
        #     noise_scale=0.05,
        # )

        # unique_fps, all_fps = fpf.find_fixed_points(state_traj, inputs)

        # fig = plot_fps(unique_fps, h_t_response)
        # fig = plot_fps(unique_fps, h_t_slices[per_idx])
        # fig = plot_fps(unique_fps, h_t.detach().cpu().numpy())


# TODO: read a yaml config for the experiment components and make a way to submit them with slurm
def submit_experiment(config_file):
    return
