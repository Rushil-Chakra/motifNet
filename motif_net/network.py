from typing import Literal, Optional, get_args

from .utils import retanh

from torch import nn
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

nonlinearity_type = Literal["softplus", "tanh", "relu", "retanh"]
init_type = Literal["diag", "gauss"]
# TODO: Implement gru
rnn_type = Literal["rnn", "gru"]


class MotifNetwork(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        nonlinearity: nonlinearity_type = "softplus",
        private_noise_std: float = 0.05,
        dt: int = 20,
        tau: int = 100,
        init_method: init_type = "diag",
        g: float = 0.8,
        seed: Optional[int] = None,
    ) -> None:
        """Model based on https://www.biorxiv.org/content/10.1101/2022.08.15.503870v1.full.pdf

        Parameters
        ---------
        hidden_size
            Number of hidden units.
        nonlinearity
            Which nonlinear function to use as activation. Possible values are
            softplus, tanh, retanh.
        private_noise_std
            Standard deviation of noise to add to input when activating (different input_array)
        dt
            Time step resolution size
        tau
            Another time step used for Euler's method of approx.
        init_method
            Method to initialize hidden weights. Either diagonal (g*I_n_hidden) or gaussian (g*N(0,1))
        seed
            Seed to use for pseudorandom generator
        """
        if nonlinearity not in get_args(nonlinearity_type):
            raise TypeError(f"nonlinearity function {nonlinearity} not valid.")
        if init_method not in get_args(init_type):
            raise TypeError(f"init type {init_type} not valid.")

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = "mps:0" if torch.backends.mps.is_available() else self.device
        logger.info(f"Device set to {self.device}")

        super(MotifNetwork, self).__init__()
        if seed is not None:
            self.rng = torch.manual_seed(seed)
        else:
            self.rng = torch.Generator()

        # Layer sizes
        self.input_size = 20
        self.hidden_size = hidden_size
        self.output_size = 3

        # Layer definitions
        self.in2hi = nn.Linear(self.input_size, self.hidden_size)
        self.hi2hi = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.hi2out = nn.Linear(self.hidden_size, self.output_size)

        self.private_noise_std = private_noise_std
        self.gamma = dt / tau

        funcs = {
            "softplus": nn.functional.softplus,
            "tanh": nn.functional.tanh,
            "relu": nn.functional.relu,
            "retanh": retanh,
        }
        self.nonlinearity = funcs[nonlinearity]

        if init_method == "diag":
            h_0_w = torch.eye((self.hidden_size)) * g
        elif init_method == "gauss":
            h_0_w = (
                torch.randn((self.hidden_size, self.hidden_size)) * g / np.sqrt(self.hidden_size)
            )
        self.hi2hi.weight = torch.nn.Parameter(h_0_w)

        # TODO: log network config

    def forward(
        self,
        x: torch.Tensor,
        h_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes a training step.

        Since an RNN trains on a sequence, each time step is shown inidividually, so that the
        output is only computed for that time step. The total output and hidden state from each time
        step are concatenated to form the outputs.

        Parameters
        ----------
        x
            The input array from a task. Take the shape (T, batch_size, 20).
        h_t
            The current hidden state of the network. This is either the initialized
            hidden state, or the output of a previous call. Takes the shape (batch_size, hidden_size)

        Returns
        -------
        y
            The estimated output. Takes shape (T, batch_shape, 3)
        h_t
            The updated hidden weights of the network. Takes the shape (T, batch_size, hidden_size)
        """
        y = torch.zeros((x.shape[0], x.shape[1], 3)).to(self.device)
        h = torch.zeros((x.shape[0], x.shape[1], self.hidden_size)).to(self.device)
        # ushing function definitions from paper
        for i in range(x.shape[0]):
            noise = torch.normal(torch.zeros(h_t.size()), self.private_noise_std).to(self.device)
            h_t = (1 - self.gamma) * h_t + self.gamma * self.nonlinearity(
                self.hi2hi(h_t) + self.in2hi(x[i]) + noise
            )
            y_t = self.hi2out(h_t)
            y[i] = y_t
            h[i] = h_t
        return y, h

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Initialize the hidden state of the network.

        Parameters
        ----------
        batch_size
            The batch size.

        Returns
        -------
        h_0
            An initialized hidden state using the ``kaiming_uniform_`` function from pytorch.
        Takes shape (batch_size, hidden_size)
        """
        h_0 = torch.nn.init.zeros_(torch.empty(batch_size, self.hidden_size)).to(self.device)
        return h_0

    # TODO: define a localized initialization where connections are in neighborhoods
