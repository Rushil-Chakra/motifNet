from motif_net.network import MotifNetwork
from motif_net.train import train
from motif_net.tasks import TaskLoader

from task_init import TASK_DICT

from omegaconf import OmegaConf

import torch
import hydra
import logging

# from FixedPointFinder.FixedPointFinderTorch import (
#     FixedPointFinderTorch as FixedPointFinder,
# )
# from FixedPointFinder.plot_utils import plot_fps

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: OmegaConf) -> None:
    logger.info("Initializing TaskLoader")
    task_list = cfg["experiments"]["task_list"]
    task_kwargs = cfg["motif_task_loader"]
    MotifTaskLoader = TaskLoader(task_dict=TASK_DICT, task_list=task_list, task_kwargs=task_kwargs)

    logger.info("Initializing network")
    network = MotifNetwork(**cfg["network"])

    logger.info("Starting training")
    model = train(network, MotifTaskLoader, **cfg["train"])

    logger.info("Saving model")
    output_path = f"{hydra.job.sweep.dir}/{hydra.job.sweep.subdir}/model.pt"
    torch.save(model, output_path)


if __name__ == "__main__":
    main()


def mlflow_log(self) -> None:
    raise NotImplementedError


def find_fixed_points(self) -> None:
    raise NotImplementedError


def plot_fixed_points(self) -> None:
    raise NotImplementedError


def submit_slurm(self) -> None:
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

#     for period_input, period_hidden_state, period in zip(
#         input_state_periods, hidden_state_periods, task.task_periods
#     ):
#         inputs, state_trajectory = fpf.sample_inputs_and_states(
#             period_input, period_hidden_state, n_inits=1000, noise_scale=0.05
#         )
#         unique_fps, all_fps = fpf.find_fixed_points(state_trajectory, inputs)
#         fig = plot_fps(unique_fps, period_hidden_state, title=period)
#         with open(f"./figures/{period}.pickle", "wb"):
#             pickle.dump(fig)

#         # inputs, state_traj = fpf.sample_inputs_and_states(
#         #     u.detach().cpu().numpy(),
#         #     h_t.detach().cpu().numpy(),
#         #     n_inits=4000,
#         #     noise_scale=0.05,
#         # )

#         # unique_fps, all_fps = fpf.find_fixed_points(state_traj, inputs)

#         # fig = plot_fps(unique_fps, h_t_response)
#         # fig = plot_fps(unique_fps, h_t_slices[per_idx])
#         # fig = plot_fps(unique_fps, h_t.detach().cpu().numpy())


# # TODO: read a yaml config for the experiment components and make a way to submit them with slurm
# def submit_experiment(config_file):
#     return
