import torch


def criterion(y: torch.Tensor, y_hat: torch.Tensor, mask: torch.Tensor) -> float:
    """Computes the loss for a model output and task.

    Parameters
    ----------
    y
        The expected output for a task instance. Takes shape (T, batch_size, 3)
    y_hat
        The output of a model evaluation. Takes shape (T, batch_shape, 3)
    mask
        The masking tensor that weighs the loss during certain time periods. Take shape (T, batch_size, 3)

    Returns
    -------
    loss
        The computed loss for the model output and task. This does not include any regularization.
    """
    loss = torch.mean(mask * torch.square(y - y_hat), dim=1).sum()
    return loss


def correct_task(theta: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """Computes accuracy of a model output.

    This is quantified by having the correct fixation output (>0.5 or =-1) and
    being within the distance threshold (< pi/10).

    Parameters
    ----------
    theta
        The expected output for a give task. Takes shape (T, batch_size, 1)
    y_hat
        The output of a model evaluation. Takes shape (T, batch_size, 3)

    Returns
    -------
    correct
        The average accuracy of the task.
    """
    theta = theta[-1]
    y_hat = y_hat[-1]
    fixate = theta < 0
    fixate_hat = y_hat[..., 0] > 0.5
    theta_hat = torch.atan2(y_hat[..., 1], y_hat[..., 2])

    dist = theta - theta_hat
    dist = torch.minimum(torch.abs(dist), 2 * torch.pi - torch.abs(dist))
    distance_check = torch.logical_and(torch.logical_not(fixate), torch.logical_not(fixate_hat))
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


def h_0_criterion(
    y: torch.Tensor, y_hat: torch.Tensor, mask: torch.Tensor, h_0: torch.Tensor
) -> float:
    """Computes the loss for a model output and task.

    Parameters
    ----------
    y
        The expected output for a task instance. Takes shape (T, batch_size, 3)
    y_hat
        The output of a model evaluation. Takes shape (T, batch_shape, 3)
    mask
        The masking tensor that weighs the loss during certain time periods. Take shape (T, batch_size, 3)

    Returns
    -------
    loss
        The computed loss for the model output and task. This does not include any regularization.
    """
    loss = torch.mean(mask * torch.square(y - y_hat), dim=1).sum()
    h_0_loss = torch.diff(h_0, dim=1).square().mean(dim=1).sum()
    loss = h_0_loss + loss
    return loss
