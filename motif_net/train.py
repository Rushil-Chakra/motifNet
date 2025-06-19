import logging

import torch
from torch import optim
import matplotlib.pyplot as plt

from .plot import plot_task, plot_performance
from .tasks import TaskLoader
from .utils import correct_task, criterion

logger = logging.getLogger(__name__)


def train(
    model: torch.nn.Module,
    output_path: str,
    task_loader: TaskLoader,
    lr: float,
    l2_terms: tuple[float, float],
    batch_size: int,
    max_iter: int = 1000000,
    clip_grad: bool = True,
    max_norm: int = 10,
):
    """Training loop for a model.

    Parameters
    ----------
    model
        Model to train
    output_path
        Directory to store model and progress plots
    task_loader
        ``TaskLoader`` object that generates data to train on
    lr
        Learning rate
    l2_terms
        l2 term for hidden weights and activation. should take form
        (l2_weight_term, l2_activation_term).
    batch_size
        Size of batches to process each step
    n_iter
        Number of training steps
    clip_grad
        Whether to clip gradients or not. Default value is True
    max_norm
        The maximum norm when clipping gradients. Default value 10
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = "mps:0" if torch.backends.mps.is_available() else device
    model.to(device)

    # Maybe make optim a hyper-parameter?
    l2_w_term = l2_terms[0]
    l2_a_term = l2_terms[1]
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_w_term)

    prev_loss = -1
    loss_arr = torch.zeros(max_iter)
    for i in range(max_iter):
        h_0 = model.init_hidden(batch_size)
        optimizer.zero_grad()

        task = task_loader.sample_task()
        x = task.get_input_array()
        y, theta = task.get_output()
        mask = task.get_mask()

        y_hat, h_t = model(x, h_0)
        activation_loss = l2_a_term * torch.square(torch.linalg.vector_norm(h_t, ord=2))
        loss = criterion(y, y_hat, mask) + activation_loss
        loss.backward()

        if i % 5000 == 0:
            accuracy_i = correct_task(theta, y_hat)
            loss_i = loss.item()
            loss_arr[i // 5000] = loss_i
            update_str = f"Progress: {i:,} Iterations\tLoss: {loss_i}\t Accuracy: {accuracy_i}"
            logger.info(update_str)

            torch.save(model, f"{output_path}/model.pt")

            task_plot, _ = plot_task(task, y_hat, i)
            task_plot.savefig(f"{output_path}/task_example.png")
            plt.close(task_plot)

            loss_plot, _ = plot_performance(loss_arr[: i // 5000], i)
            loss_plot.savefig(f"{output_path}/loss_plot.png")
            plt.close(loss_plot)

        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if loss.item() == prev_loss:
            logger.info("loss stabilized. Stopping training.")
            break

        prev_loss = loss.item()

    logger.info("training complete")
    return model
