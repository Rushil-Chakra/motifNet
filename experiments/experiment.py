import logging
import pickle
import sys
import traceback

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from motif_net.task_init import TASK_DICT

from motif_net.FixedPointFinder import FixedPointFinderTorch as FixedPointFinder
from motif_net.FixedPointFinder import plot_fps
from motif_net.network import MotifNetwork
from motif_net.tasks import TaskLoader
from motif_net.train import train

# from FixedPointFinder.FixedPointFinderTorch import (
#     FixedPointFinderTorch as FixedPointFinder,
# )
# from FixedPointFinder.plot_utils import plot_fps

logger = logging.getLogger(__name__)


def get_tasks_periods(
    model: MotifNetwork, task_loader: TaskLoader, task_name: str, batch_size: int
) -> tuple[list, list, list]:
    task = task_loader.init_task(task_name)
    input = task.get_input_array()
    h_0 = model.init_hidden(batch_size)
    with torch.no_grad():
        _, h_t = model(input, h_0)

    input_state_periods = [
        period.input_array.detach().cpu().numpy() for period in task.task_periods
    ]

    task_times = task.get_time_spans()
    hidden_state_periods = [
        h_t[period_times["start"] : period_times["end"]].cpu().detach().numpy()
        for period_times in task_times.values()
    ]
    task_periods = [period.period_name for period in task.task_periods]
    return input_state_periods, hidden_state_periods, task_periods


def find_fixed_points(
    model: MotifNetwork,
    input_state_periods: list,
    hidden_state_periods: list,
    task_periods: list,
    fpf_hp: dict = {},
):
    fpf = FixedPointFinder(model, **fpf_hp)
    for period_input, period_hidden_state, period in zip(
        input_state_periods, hidden_state_periods, task_periods
    ):
        logger.info(f"Finding fixed points for {period}...")
        inputs, state_trajectory = fpf.sample_inputs_and_states(
            period_input, period_hidden_state, n_inits=1000, noise_scale=0.05
        )
        unique_fps, all_fps = fpf.find_fixed_points(state_trajectory, inputs)
        fig = plot_fps(unique_fps, period_hidden_state, title=period)
        with open(
            f"{HydraConfig.get().sweep.dir}/{HydraConfig.get().sweep.subdir}/{period}.pickle", "wb"
        ) as file:
            logger.info(f"Writing matplotlib fig {period}")
            pickle.dump(fig, file)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: OmegaConf) -> None:
    try:
        logger.info("\n" + OmegaConf.to_yaml(cfg))
        logger.info(f"Using device {torch.cuda.get_device_name()}")
        logger.info("Initializing TaskLoader")
        task_list = cfg["experiments"]["task_list"]
        task_kwargs = cfg["task_loader"]
        MotifTaskLoader = TaskLoader(
            task_dict=TASK_DICT, task_list=task_list, task_kwargs=task_kwargs
        )

        logger.info("Initializing network")
        network = MotifNetwork(**cfg["network"])

        output_path = f"{HydraConfig.get().sweep.dir}/{HydraConfig.get().sweep.subdir}"

        logger.info("Starting training")
        model = train(network, output_path, MotifTaskLoader, **cfg["train"])

        model_path = f"{output_path}/model.pt"
        logger.info(f"Saving model to {model_path}")
        torch.save(model, model_path)

        # model = torch.load(
        #     "../models/runs/2025-05-07/23-26-29/experiments=ReactPro/model.pt", weights_only=False
        # )
        logger.info("Starting fixed point finding")
        batch_size = cfg["task_loader"]["batch_size"]
        task_kwargs["gamma"] = None
        FPTaskLoader = TaskLoader(
            task_dict=TASK_DICT, task_list=task_list, task_kwargs=task_kwargs
        )
        input_state_periods, hidden_state_periods, task_periods = get_tasks_periods(
            model, FPTaskLoader, task_list[0], batch_size
        )
        find_fixed_points(model, input_state_periods, hidden_state_periods, task_periods)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()


# What do I need logged?
# could use for logging the experiment specific stuff and graphs
def mlflow_log(self) -> None:
    raise NotImplementedError


def plot_fixed_points(self) -> None:
    raise NotImplementedError

    # def find_fixed_points(trained_model, batch_size):
    #     # Get the trained model's hidden weights and input
    #     task = task_init.MemoryPro(batch_size=batch_size)
    #     u = task.get_input_array()
    #     h_t = trained_model.init_hidden(batch_size)
    #     with torch.no_grad():
    #         output, h_t = trained_model(u, h_t)

    #     # TODO: Define why im doing this, I know it has to do with slicing weights on time period
    #     # to get separate dynamics for each trial period
    #     def find_fixed_points_period(h_t, period_times):
    #         h_t_slice = (
    #             h_t[period_times["start"] : period_times["end"]].cpu().detach().numpy()
    #         )
    #         return h_t_slice

    #     task_times = task.get_time_spans()
    #     input_state_periods = [
    #         period.input_array.detach().cpu().numpy() for period in task.task_periods
    #     ]
    #     hidden_state_periods = [
    #         find_fixed_points_period(h_t, period_times)
    #         for period_times in task_times.values()
    #     ]
    #     fpf = FixedPointFinder(
    #         trained_model,
    #         method="joint",
    #         tol_q=1e-9,
    #         max_iters=10000,
    #         outlier_distance_scale=50.0,
    #     )

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
