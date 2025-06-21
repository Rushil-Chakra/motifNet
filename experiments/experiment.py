import logging
import sys
import traceback

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from motif_net.task_init import TASK_DICT

from motif_net.find_fixed_points import get_task_periods, find_fixed_points
from motif_net.network import MotifNetwork
from motif_net.tasks import TaskLoader
from motif_net.train import train


logger = logging.getLogger(__name__)


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

        logger.info("Starting fixed point finding")
        batch_size = 50
        task_kwargs["gamma"] = None
        for task_name in task_list:
            task = TASK_DICT[task_name](batch_size=task_kwargs["batch_size"], circle=True)

            input_state_periods, hidden_state_periods, output_state_periods, task_periods = (
                get_task_periods(model, task, batch_size)
            )
            fig, unique_fixed_points = find_fixed_points(
                model,
                input_state_periods,
                hidden_state_periods,
                output_state_periods,
                task_periods,
                output_path,
            )
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
