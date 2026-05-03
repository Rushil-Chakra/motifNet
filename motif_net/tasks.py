from typing import Literal, Optional, get_args

from dataclasses import dataclass

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


@dataclass(repr=False)
class TaskPeriod:
    period_name: period_name_type
    task_name: str
    min_t: int
    max_t: int
    dt: int = 20
    batch_size: int = 1

    def __post_init__(self) -> None:
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
        """
        if self.period_name not in get_args(period_name_type):
            raise ValueError("Period name not found.")
        if self.task_name not in TASK_VECTOR_DEFINITION:
            raise ValueError("Task name not found")

        # Gets an int instead of int-typed Tuple
        self.n_steps = (
            (torch.randint(self.min_t, self.max_t, (1,)) / self.dt).int().item()
        )
        fixation = 0 if self.period_name == "response" else 1

        self.input_array = torch.zeros((self.n_steps, self.batch_size, 20))
        self.input_array[..., 0] = fixation

        task_one_hot_idx = 5 + TASK_VECTOR_DEFINITION.index(self.task_name)
        self.input_array[..., task_one_hot_idx] = 1

    def __repr__(self):
        return f"{self.period_name} | n_steps: {self.n_steps}"

    def set_stimulus_values(
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
        # Makes this nice for when I need to set 2 modalities at once
        if isinstance(amplitude, float):
            amplitude = torch.Tensor([amplitude]).unsqueeze(1)
        mod_idx = 2 * modality - 1
        offset = n_mod * 2
        # gotta do this to set for each sample in batch
        # Modalities start at 1, assuming only 2 values possible
        stim_vals = []
        for i in range(n_mod):
            stim_vals.append(
                torch.stack(
                    (
                        amplitude[:, i] * torch.sin(theta[:, i]),
                        amplitude[:, i] * torch.cos(theta[:, i]),
                    ),
                    dim=1,
                )
            )
        stim_vals = torch.cat(stim_vals, dim=1)

        for i in range(self.input_array.shape[1]):
            self.input_array[:, i, mod_idx[i] : mod_idx[i] + offset] = stim_vals[i, :]


class Task:
    def __init__(
        self,
        name: str,
        batch_size: int = 1,
        dt: int = 20,
        gamma: Optional[float] = None,
    ) -> None:
        # TODO: Add more to docstring
        """A collection of task periods meant to represent a single task input.

        Each instance creates an object capable of generating random batches of trials.

        Note
        ----
        Any class function that is prefixed with ``generate_`` denotes a stochastic function
        that will have different values each call. A function prefixed with ``get_`` denotes
        a deterministic function that will have the same return value each call.

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
        """
        if name not in TASK_VECTOR_DEFINITION:
            raise ValueError("Task name not found.")
        self.name = name
        self.batch_size = batch_size
        self.task_periods = []
        self.dt = dt
        self.gamma = gamma

        # These get set after a call to set_task_periods
        self.n_steps = None
        self.y = None
        self.theta = None

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = "mps:0" if torch.backends.mps.is_available() else device
        self.device = device

    def __repr__(self):
        name_str = (
            f"{self.name} - batch_size: {self.batch_size} - n_steps: {self.n_steps}\n"
        )
        for task_period in self.task_periods:
            name_str += f"{task_period}\n"
        return name_str

    def generate_stimulus_batch(
        self, n_mod: int = 1, circle: bool = False
    ) -> torch.Tensor:
        """Create a batch of angles in radians as an input for a task period.

        Parameters
        ----------
        n_mod
            The number of modalities to generate thetas for.
        circle
            Boolean to say whether to randomly generate theta values or set them to be evenly
            spaced points around a circle. The default value is ``False``.

        Returns
        -------
        theta
            A batch of angles to add to an ``input_array`` for a task period. Takes shape
            (batch_size, n_mod)
        """
        if circle:
            points = torch.linspace(0, 2 * torch.pi, self.batch_size + 1)[
                :-1
            ].unsqueeze(1)
            theta = theta = torch.ones((self.batch_size, n_mod)) * points
        else:
            theta = torch.rand((self.batch_size, n_mod)) * 2 * torch.pi
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
        mask = mask.to(self.device)
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
        self.y = torch.zeros((self.n_steps, self.batch_size, 3))
        # Setting fixation periods
        response_start = self.n_steps - self.task_periods[-1].n_steps
        self.y[:response_start, :, 0] = 1

        # -1 is fixation location required when no stimulus is presented
        # Or if the network should not move to stimulus
        self.theta = -torch.ones((self.n_steps, self.batch_size))

    def set_y(self) -> None:
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
                torch.randn(task_input_array.size()) * np.sqrt(2 / self.gamma) * 0.1
            )
        task_input_array = task_input_array.to(self.device)
        return task_input_array

    def get_output(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the outputs needed for network computation.

        Returns
        -------
        y
            The expected output that a network must match.
            Takes shape (T, batch_size, 3)
        theta
            The thetas that a network must match. Easier to have
            than to compute from y. Takes shape (T, batch_size)"""
        y, theta = self.y.to(self.device), self.theta.to(self.device)
        return y, theta

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

    def generate_task_period(
        self,
        period_name: str,
        min_t: int,
        max_t: int,
        stimulus_kwargs: Optional[dict] = None,
    ) -> TaskPeriod:
        """Instantiate a TaskPeriod object to add to ``task_periods``.

        This helps get rid of some boilerplates. Once a set of task_periods are created,
        use ``add_task_periods`` to add them back to this object and instantiate the inputs/outputs

        Parameters
        ----------
        period_name
            Name of the `TaskPeriod`. Certain periods have specific properties.
        min_t
            Minimum bound for randomly generated time span.
        max_t
            Maximum bound for randomly generated time span.
        stimulus_kwargs
            Named arguments for creating inputs/outputs for the stimulus time periods.

        Returns
        -------
        task_period
            The instantiated TaskPeriod object
        """
        task_period = TaskPeriod(
            period_name,
            self.name,
            min_t=min_t,
            max_t=max_t,
            dt=self.dt,
            batch_size=self.batch_size,
        )
        if stimulus_kwargs is not None:
            task_period.set_stimulus_values(**stimulus_kwargs)

        return task_period


class TaskLoader:
    def __init__(
        self,
        task_dict: dict,
        task_list: Optional[list] = None,
        task_probs: Optional[torch.Tensor] = None,
        task_kwargs: dict = {},
    ) -> None:
        """Class to initialize specific ``Task`` objects or randomly sample with given probabilities.

        Parameters
        ----------
        task_dict
            Dictionary where the keys are names of the tasks and values are ``Task`` objects. Used
            for indexing specific objects with a string.
        task_list, optional
            List of task names to draw random samples from, by default None
        task_probs, optional
            Tensor of probabilities of drawing each ``Task``. By default, probabilities are uniform.
        task_kwargs, optional
            Named parameters to pass to the ``Task`` object, by default ``{}``
        """
        self.task_dict = task_dict
        self.task_kwargs = task_kwargs

        if task_list is None:
            task_list = list(self.task_dict.keys())
        if task_probs is None:
            task_probs = torch.ones(len(task_list)) / len(task_list)

        self.task_list = task_list
        self.task_probs = task_probs

    def init_task(self, task_name: str) -> Task:
        """Initialize a ``Task``

        Parameters
        ----------
        task_name
            Name of ``Task``

        Returns
        -------
        task_obj
            An instantiation of the task object requested
        """
        task_obj = self.task_dict[task_name](**self.task_kwargs)
        return task_obj

    def sample_task(
        self,
    ) -> Task:
        """Randomly picks a task from a given list.

        Returns
        -------
        task
            The sampled ``Task`` object.
        """
        idx = torch.multinomial(self.task_probs, 1, replacement=True).item()
        task = self.init_task(self.task_list[idx])

        return task
