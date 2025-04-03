from .tasks import TaskLoader

from .utils import criterion, correct_task

import torch
from torch import optim

# from torch.utils.tensorboard import SummaryWriter

import logging

logger = logging.getLogger(__name__)


def train(
    model: torch.nn.Module,
    task_loader: TaskLoader,
    lr: float,
    l2_terms: tuple[float, float],
    batch_size: int,
    max_iter: int = 50000000,
    clip_grad: bool = True,
    max_norm: int = 10,
):
    """Training loop for a model.

    Parameters
    ----------
    model
        Model to train
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
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    model.to(device)

    # writer = SummaryWriter()

    # Maybe make optim a hyper-parameter?
    l2_w_term = l2_terms[0]
    l2_a_term = l2_terms[1]
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_w_term)

    prev_loss = -1
    for i in range(max_iter):
        h_t = model.init_hidden(batch_size)
        optimizer.zero_grad()

        task = task_loader.sample_task()
        x = task.get_input_array()
        y, theta = task.get_output()
        mask = task.get_mask()

        y_hat, h_t = model(x, h_t)
        activation_loss = l2_a_term * torch.square(torch.norm(h_t, p=2))
        loss = criterion(y, y_hat, mask) + activation_loss
        loss.backward()

        if i % 10000 == 0:
            accuracy_i = correct_task(theta, y_hat)
            loss_i = loss.item()
            # fig1, _ = plot.stim_plotter(
            #     theta.detach().cpu().numpy(), z_hat.detach().cpu().numpy()
            # )
            # fig2, _ = plot.trial_plotter(
            #     z.detach().cpu().numpy(),
            #     z_hat.detach().cpu().numpy(),
            #     task,
            # )
            # fig3, _ = plot.pca_plotter(hidden_state=h_t.detach().cpu().numpy())

            # writer.add_scalar("Loss", loss_i, i)
            # writer.add_scalar("Accuracy", accuracy_i, i)
            # writer.add_figure("Plot", fig2, i)
            update_str = f"Progress: {i/max_iter:.2%}\tLoss: {loss_i}\t Accuracy: {accuracy_i}"
            logger.info(update_str)

        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if loss.item() == prev_loss:
            logger.info("loss stabilized. Stopping training.")
            break

        prev_loss = loss.item()

    # writer.flush()
    return model
