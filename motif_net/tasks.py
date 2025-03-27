from typing import Literal, Optional, get_args

import numpy as np

import torch

TASK_VECTOR_DEFINITION = [
    "DelayedPro",
    "DelayedAnti",
    "MemoryPro",
    "MemoryAnti",
    "ReactPro",
    "ReactAnti",
    "IntegrationModality1",
    "IntegrationModality2",
    "ContextIntModality1",
    "ContextIntModality2",
    "IntegrationMultiModal",
    "ReactMatch2Sample",
    "ReactNonMatch2Sample",
    "ReactCategoryPro",
    "ReactCategoryAnti",
]

period_name_type = Literal[
    "context", "stimulus1", "stimulus2", "memory1", "memory2", "response"
]


class TaskPeriod:
    def __init__(
        self,
        period_name: period_name_type,
        task_name: str,
        min_t: int,
        max_t: int,
        dt: int = 20,
        batch_size: int = 1,
        rng: torch.Generator = None,
    ) -> None:
        """Definition for each task period.

        A set of task periods make up a single ``Task``.

        Parameters
        ----------
        period_name
            Name of the `TaskPeriod`. Certain periods have specific properties.
        task_name
            Name of the ``Task`` the ``TaskPeriod`` is part of.
        min_t
            Minimum bound for randomly generated time span.
        max_t
            Maximum bound for randomly generated time span.
        dt
            Size of time step.
        batch_size
            Number of trials per batch.
        rng
            Torch generator object used to generate pseudorandom values.
        """
        if period_name not in get_args(period_name_type):
            raise ValueError("Period name not found.")
        if task_name not in TASK_VECTOR_DEFINITION:
            raise ValueError("Task name not found")
        self.period_name = period_name
        self.rng = rng

        # Gets an int instead of int-typed Tuple
        self.n_steps = (
            (torch.randint(min_t, max_t, (1,), generator=rng) / dt).int().item()
        )
        fixation = 0 if period_name == "response" else 1

        self.input_array = torch.zeros((self.n_steps, batch_size, 20))
        self.input_array[..., 0] = fixation

        task_one_hot_idx = 5 + TASK_VECTOR_DEFINITION.index(task_name)
        self.input_array[..., task_one_hot_idx] = 1

    def __repr__(self):
        return f"{self.period_name} | n_steps: {self.n_steps}"

    def add_stimulus_values(
        self,
        theta: torch.Tensor,
        modality: torch.Tensor,
        amplitude: float | torch.Tensor = 1.0,
        n_mod: int = 1,
    ) -> None:
        """Take a batch of angles representing a stimulus and add them to the ``input_array``.

        Parameters
        ----------
        theta
            Batch of angles in radians to set as the stimulus value for a stimulus time period.
            For a given set of angles, the input is set to be [Asin(theta), Acos(theta)] for each
            modality present where A is the ``amplitude``. Should take shape (n_mod, batch_size).
        modality
            The modality to set each angle at. Modality values are either 1 or 2, and correspond
            to 1:3 or 3:5 of the input nodes. Should take shape (batch_size, n_mod).
        amplitude
            The strength of the stimulus. Passing a float will change the amplitude of all stimulus
            values by the same amount. Passing a tensor will modify each angle individually. Should
            take shape (batch_size, n_mod)
        n_mod
            The number of stimulus modalities being added. Should be only either 1 or 2.
        """
        mod_idx = 2 * modality - 1
        offset = n_mod * 2
        # gotta do this to set for each sample in batch
        # Modalities start at 1, assuming only 2 values possible
        stim_vals = torch.cat(
            (amplitude * torch.sin(theta), amplitude * torch.cos(theta)), dim=1
        )
        for i in range(self.input_array.shape[1]):
            self.input_array[:, i, mod_idx[i] : mod_idx[i] + offset] = stim_vals[i, :]


class Task:
    def __init__(
        self,
        name: str,
        batch_size: int = 64,
        dt: int = 20,
        gamma: Optional[float] = None,
        rng: torch.Generator = None,
    ) -> None:
        """A collection of task periods meant to represent a single task input.

        Each instance creates an object capable of generating random batches of trials.

        Parameters
        ----------
        name
            Name of the task. Must appear in ``TASK_VECTOR_DEFINITION``.
        batch_size
            The number of trials per batch.
        dt
            Time step resolution. Times ranges from the paper are divided
            by this value.
        gamma
            Equivalent to dt/tau in paper. Used here to add private noise to the input.
            If set to None, it does not add any noise.
        rng
            Torch generator object used to generate pseudorandom values.
        """
        if name not in TASK_VECTOR_DEFINITION:
            raise ValueError("Task name not found.")
        self.name = name
        self.batch_size = batch_size
        self.task_periods = []
        self.dt = dt
        self.gamma = gamma
        self.rng = rng

        # These get set after a call to set_task_periods
        self.n_steps = None
        self.z = None
        self.theta = None

    def __repr__(self):
        return f"{self.name} | n_steps: {self.n_steps}"

    def generate_stimulus_batch(self, n_mod=1) -> torch.Tensor:
        """Create a batch of angles in radians as an input for a task period.

        Parameters
        ----------
        n_mod
            The number of modalities to generate thetas for.

        Returns
        -------
        theta
            A batch of angles to add to an ``input_array`` for a task period. Takes shape
            (batch_size, n_mod)
        """
        theta = torch.rand((self.batch_size, n_mod), generator=self.rng) * 2 * torch.pi
        return theta

    def generate_modality(self) -> torch.Tensor:
        """Creates a batch of modalities to set thetas for a task period.

        Returns
        -------
        modality
            A batch of modalities. Takes shape (batch_size, 1). Values are either 1 or 2.
        """
        modality = torch.randint(1, 3, (self.batch_size,))
        return modality

    def get_mask(self) -> torch.Tensor:
        """Create a mask tensor used for adding weight to the output when computing the loss function.

        Returns
        -------
        mask
            A tensor of shape (T, batch_size, 3) where T is the length of the trial. Values are 0 during
            grace periods, 1 during fixation, and 5 during response. These values are doubled for the fixation
            output unit (0 index in the 3rd dimension).
        """
        mask = torch.ones((self.n_steps, self.batch_size, 3))
        delay = int(100 / self.dt)
        mask[:delay] = 0

        # React tasks don't have a named response period, but the last task in list is response
        response_start = self.n_steps - self.task_periods[-1].n_steps
        mask[response_start : response_start + delay] = 0
        mask[response_start + delay :] = 5
        # The context output has double the penalty
        mask[..., 0] *= 2
        return mask

    def set_task_periods(self, task_periods: list) -> None:
        """Add a list of task periods to the ``Task`` and instantiate input and output arrays based on it.

        Parameters
        ----------
        task_periods
            List of ``TaskPeriods`` that make up the task.
        """
        self.task_periods = task_periods
        self.n_steps = sum([period.n_steps for period in self.task_periods])
        self.z = torch.zeros((self.n_steps, self.batch_size, 3))
        # Setting fixation periods
        response_start = self.n_steps - self.task_periods[-1].n_steps
        self.z[:response_start, :, 0] = 1

        # -1 is fixation location required when no stimulus is presented
        # Or if the network should not move to stimulus
        self.theta = -torch.ones((self.n_steps, self.batch_size))

    def set_z(self) -> None:
        """Custom function per task to create the output array.

        Each task has different requirements, so the way that the output array is created differs for each
        task. However all will take the shape (T, batch_size, 3)
        """
        raise NotImplementedError

    def get_input_array(self) -> torch.Tensor:
        """Get the input_array by concatenating each individual task ``input_array``.
        Also adds noise if ``self.gamma`` is set.

        Returns
        -------
        task_input_array
            The input array for the task. Takes shape (T, batch_size, 20).
        """
        task_input_array = torch.cat(
            [period.input_array for period in self.task_periods], dim=0
        )
        if self.gamma is not None:
            task_input_array += (
                torch.randn(task_input_array.size(), generator=self.rng)
                * np.sqrt(2 / (self.gamma))
                * 0.1
            )
        return task_input_array

    def get_output(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the outputs needed for network computation.

        Returns
        -------
        z
            The expected output that a network must match.
            Takes shape (T, batch_size, 3)
        theta
            The thetas that a network must match. Easier to have
            than to compute from z. Takes shape (T, batch_size)"""
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # device = "mps:0" if torch.backends.mps.is_available() else device
        z, theta = self.z.to(device), self.theta.to(device)
        return z, theta

    def get_time_spans(self) -> dict[str, dict[str, int]]:
        """Returns a dictionary with the start and end time of each task period.

        Returns
        -------
        time_spans
            Dictionary taking form ``{task_period.name: {"start": start_time, "end": end_time}}
            Note that these are half-open intervals, so each time period spans [start_time, end_time).``
        """
        total_time = 0
        # Walrus operator evauluates the right hand side as well as saves it, letting total_time become
        # A running total
        time_spans = {
            period.period_name: {
                "start": total_time,
                "end": (total_time := total_time + period.n_steps),
            }
            for period in self.task_periods
        }
        return time_spans
