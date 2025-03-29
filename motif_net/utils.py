from typing import Tuple

import torch
import yaml

def criterion(z: torch.Tensor, z_hat: torch.Tensor, mask: torch.Tensor) -> float:
    """Computes the loss for a model output and task.

    Parameters
    ----------
    z
        The expected output for a task instance. Takes shape (T, batch_size, 3)
    z_hat
        The output of a model evaluation. Takes shape (T, batch_shape, 3)
    mask
        The masking tensor that weighs the loss during certain time periods. Take shape (T, batch_size, 3)

    Returns
    -------
    loss
        The computed loss for the model output and task. This does not include any regularization.
    """
    loss = torch.mean(mask * torch.square(z - z_hat))
    return loss


# TODO: Figure out how htis factors into the training
def correct_task(theta: torch.Tensor, z_hat: torch.Tensor) -> torch.Tensor:
    """Computes accuracy of a model output.

    This is quantified by having the correct fixation output (>0.5 or =-1) and
    being within the distance threshold (< pi/10).

    Parameters
    ----------
    theta
        The expected output for a give task. Takes shape (T, batch_size, 1)
    z_hat
        The output of a model evaluation. Takes shape (T, batch_size, 3)

    Returns
    -------
    correct
        The average accuracy of the task.
    """
    theta = theta[-1]
    z_hat = z_hat[-1]
    fixate = theta < 0
    fixate_hat = z_hat[..., 0] > 0.5
    theta_hat = torch.atan2(z_hat[..., 1], z_hat[..., 2])

    dist = theta - theta_hat
    dist = torch.minimum(torch.abs(dist), 2 * torch.pi - torch.abs(dist))
    distance_check = torch.logical_and(
        torch.logical_not(fixate), torch.logical_not(fixate_hat)
    )
    distance_check = torch.logical_and(distance_check, dist < torch.pi / 10)
    fixation_check = torch.logical_and(fixate, fixate_hat)

    correct = torch.logical_or(distance_check, fixation_check).float().mean()
    return correct


def retanh(input: torch.Tensor) -> torch.Tensor:
    """Relu(tanh).
    Parameters
    ----------
    input
        Input torch.tensor. Expecting size (T, batch_size, 20).

    Returns
    -------
    output
        Retanh of the input.
    """
    tanh = torch.nn.functional.tanh(input)
    output = torch.nn.functional.relu(tanh)
    return output


def load_config(yaml_path: str) -> Tuple[dict, dict]:
    """_summary_

    Parameters:
    -----------
        yaml_path
            Path to yaml config file.

    Returns:
    --------
        train_config
            Parameters for training configuration 
        model_config
            Parameters for model configuration
    """
    with open(yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    train_config = config["train"]
    model_config = config["model"]
    
    return train_config, model_config