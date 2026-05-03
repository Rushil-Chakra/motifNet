from dataclasses import dataclass
from typing import Optional

import torch

from .tasks import Task


@dataclass(repr=False)
class DelayedPro(Task):
    batch_size: int = 1
    dt: int = 20
    gamma: float | None = None
    circle: bool = False

    def __post_init__(self) -> None:
        super().__init__("DelayedPro", self.batch_size, self.dt, self.gamma)

        theta = self.generate_stimulus_batch(circle=self.circle)
        modality = self.generate_modality()
        stim_kwargs = {"theta": theta, "modality": modality}

        context = self.generate_task_period("context", 300, 700)
        stim_1 = self.generate_task_period("stimulus1", 300, 1500, stim_kwargs)
        response = self.generate_task_period("response", 300, 1500, stim_kwargs)

        self.set_task_periods([context, stim_1, response])
        self.set_y(theta)

    def set_y(self, theta: torch.Tensor) -> None:
        """Move to same angle as presented."""
        response_start = self.n_steps - self.task_periods[-1].n_steps
        self.y[response_start:, :, 1:] = torch.cat(
            (torch.sin(theta), torch.cos(theta)), dim=1
        )
        self.theta[response_start:] = theta.squeeze(1)


@dataclass(repr=False)
class DelayedAnti(Task):
    batch_size: int = 1
    dt: int = 20
    gamma: float | None = None
    circle: bool = False

    def __post_init__(self) -> None:
        super().__init__("DelayedAnti", self.batch_size, self.dt, self.gamma)
        theta = self.generate_stimulus_batch(circle=self.circle)
        modality = self.generate_modality()
        stimulus_kwargs = {"theta": theta, "modality": modality}

        context = self.generate_task_period("context", 300, 700)
        stim_1 = self.generate_task_period("stimulus1", 300, 1500, stimulus_kwargs)
        response = self.generate_task_period("response", 300, 1500, stimulus_kwargs)

        self.set_task_periods([context, stim_1, response])
        self.set_y(theta)

    def set_y(self, theta: torch.Tensor) -> None:
        """Move in opposite direction of angle presented."""
        theta += torch.pi
        response_start = self.n_steps - self.task_periods[-1].n_steps
        self.y[response_start:, :, 1:] = torch.cat(
            (torch.sin(theta), torch.cos(theta)), dim=1
        )
        self.theta[response_start:] = theta.squeeze(1)


@dataclass(repr=False)
class MemoryPro(Task):
    batch_size: int = 1
    dt: int = 20
    gamma: float | None = None
    circle: bool = False

    def __post_init__(self) -> None:
        super().__init__("MemoryPro", self.batch_size, self.dt, self.gamma)

        theta = self.generate_stimulus_batch(circle=self.circle)
        modality = self.generate_modality()
        stimulus_kwargs = {"theta": theta, "modality": modality}

        context = self.generate_task_period("context", 300, 700)
        stim_1 = self.generate_task_period("stimulus1", 200, 1600, stimulus_kwargs)
        memory_1 = self.generate_task_period("memory1", 200, 1600)
        response = self.generate_task_period("response", 300, 700)

        self.set_task_periods([context, stim_1, memory_1, response])
        self.set_y(theta)

    def set_y(self, theta: torch.Tensor) -> None:
        """Move in same angle as presented after stimulus has faded."""
        response_start = self.n_steps - self.task_periods[-1].n_steps
        self.y[response_start:, :, 1:] = torch.cat(
            (torch.sin(theta), torch.cos(theta)), dim=1
        )
        self.theta[response_start:] = theta.squeeze(1)


@dataclass(repr=False)
class MemoryAnti(Task):
    batch_size: int = 1
    dt: int = 20
    gamma: float | None = None
    circle: bool = False

    def __post_init__(self) -> None:
        super().__init__("MemoryAnti", self.batch_size, self.dt, self.gamma)

        theta = self.generate_stimulus_batch(circle=self.circle)
        modality = self.generate_modality()
        stimulus_kwargs = {"theta": theta, "modality": modality}

        context = self.generate_task_period("context", 300, 700)
        stim_1 = self.generate_task_period("stimulus1", 200, 1600, stimulus_kwargs)
        memory_1 = self.generate_task_period("memory1", 200, 1600)
        response = self.generate_task_period("response", 300, 700)

        self.set_task_periods([context, stim_1, memory_1, response])
        self.set_y(theta)

    def set_y(self, theta: torch.Tensor) -> None:
        """Move in opposite direction of stimulus after it has faded."""
        theta += torch.pi
        response_start = self.n_steps - self.task_periods[-1].n_steps
        self.y[response_start:, :, 1:] = torch.cat(
            (torch.sin(theta), torch.cos(theta)), dim=1
        )
        self.theta[response_start:] = theta.squeeze(1)


@dataclass(repr=False)
class ReactPro(Task):
    batch_size: int = 1
    dt: int = 20
    gamma: float | None = None
    circle: bool = False

    def __post_init__(self) -> None:
        super().__init__("ReactPro", self.batch_size, self.dt, self.gamma)

        theta = self.generate_stimulus_batch(circle=self.circle)
        modality = self.generate_modality()
        stimulus_kwargs = {"theta": theta, "modality": modality}

        context = self.generate_task_period("context", 500, 2500)
        # Response period as well, but react tasks have fixation = 1 even during response
        stim_1 = self.generate_task_period("stimulus1", 300, 1700, stimulus_kwargs)

        self.set_task_periods([context, stim_1])
        self.set_y(theta)

    def set_y(self, theta: torch.Tensor) -> None:
        """Move in direction of stimulus immediately. There is no change in fixation during
        the whole trial."""
        response_start = self.n_steps - self.task_periods[-1].n_steps
        self.y[response_start:, :, 1:] = torch.cat(
            (torch.sin(theta), torch.cos(theta)), dim=1
        )
        self.theta[response_start:] = theta.squeeze(1)


@dataclass(repr=False)
class ReactAnti(Task):
    batch_size: int = 1
    dt: int = 20
    gamma: float | None = None
    circle: bool = False

    def __post_init__(self) -> None:
        super().__init__("ReactAnti", self.batch_size, self.dt, self.gamma)

        theta = self.generate_stimulus_batch(circle=self.circle)
        modality = self.generate_modality()
        stimulus_kwargs = {"theta": theta, "modality": modality}

        context = self.generate_task_period("context", 500, 2500)
        # Response period as well, but react tasks have fixation = 1 even during response
        stim_1 = self.generate_task_period("stimulus1", 300, 1700, stimulus_kwargs)

        self.set_task_periods([context, stim_1])
        self.set_y(theta)

    def set_y(self, theta):
        """Move in direction of stimulus immediately. There is no change in fixation during
        the whole trial."""
        theta += torch.pi
        response_start = self.n_steps - self.task_periods[-1].n_steps
        self.y[response_start:, :, 1:] = torch.cat(
            (torch.sin(theta), torch.cos(theta)), dim=1
        )
        self.theta[response_start:] = theta.squeeze(1)


@dataclass(repr=False)
class IntegrationModality1(Task):
    batch_size: int = 1
    dt: int = 20
    gamma: float | None = None
    circle: bool = False

    def __post_init__(self) -> None:
        super().__init__("IntegrationModality1", self.batch_size, self.dt, self.gamma)

        # Using values from code
        amp_mean = 0.4 * torch.rand((self.batch_size, 1)) + 0.8
        amp_cohesion_vars = torch.Tensor([0.08, 0.16, 0.32])
        amp_cohesion_sign = torch.Tensor([-1, 1])
        amp_var = (
            amp_cohesion_vars[torch.randint(0, 3, (self.batch_size, 1))]
            * amp_cohesion_sign[torch.randint(0, 2, (self.batch_size, 1))]
        )
        amp_1 = amp_mean + amp_var
        amp_2 = amp_mean - amp_var

        theta_1 = self.generate_stimulus_batch(circle=self.circle)
        theta_2 = self.generate_stimulus_batch(circle=self.circle)
        modality = torch.ones((self.batch_size,), dtype=int)

        stimulus1_kwargs = {"theta": theta_1, "modality": modality, "amplitude": amp_1}
        stimulus2_kwargs = {"theta": theta_2, "modality": modality, "amplitude": amp_2}

        context = self.generate_task_period("context", 200, 600)
        stim_1 = self.generate_task_period("stimulus1", 200, 1600, stimulus1_kwargs)
        memory_1 = self.generate_task_period("memory1", 200, 1600)
        stim_2 = self.generate_task_period("stimulus2", 200, 1600, stimulus2_kwargs)
        memory_2 = self.generate_task_period("memory2", 100, 300)
        response = self.generate_task_period("response", 300, 700)

        self.set_task_periods([context, stim_1, memory_1, stim_2, memory_2, response])
        self.set_y(theta_1, theta_2, amp_1, amp_2)

    def set_y(
        self,
        theta_1: torch.Tensor,
        theta_2: torch.Tensor,
        amp_1: torch.Tensor,
        amp_2: torch.Tensor,
    ) -> None:
        """Move to stimulus with largest amplitude. Only modality 1 presented"""
        amp = torch.cat((amp_1, amp_2), dim=1)
        amp_idx = torch.argmax(amp, dim=1)

        theta = torch.cat((theta_1, theta_2), dim=1)
        theta = torch.gather(theta, 1, amp_idx.unsqueeze(1))

        response_start = self.n_steps - self.task_periods[-1].n_steps

        self.y[response_start:, :, 1:] = torch.cat(
            (torch.sin(theta), torch.cos(theta)), dim=1
        )
        self.theta[response_start, :] = theta.squeeze(1)


@dataclass(repr=False)
class IntegrationModality2(Task):
    batch_size: int = 1
    dt: int = 20
    gamma: float | None = None
    circle: bool = False

    def __post_init__(self) -> None:
        super().__init__("IntegrationModality2", self.batch_size, self.dt, self.gamma)

        # Using values from code
        amp_mean = 0.4 * torch.rand((self.batch_size, 1)) + 0.8
        amp_cohesion_vars = torch.Tensor([0.08, 0.16, 0.32])
        amp_cohesion_sign = torch.Tensor([-1, 1])
        amp_var = (
            amp_cohesion_vars[torch.randint(0, 3, (self.batch_size, 1))]
            * amp_cohesion_sign[torch.randint(0, 2, (self.batch_size, 1))]
        )
        amp_1 = amp_mean + amp_var
        amp_2 = amp_mean - amp_var

        theta_1 = self.generate_stimulus_batch(circle=self.circle)
        theta_2 = self.generate_stimulus_batch(circle=self.circle)
        modality = torch.ones((self.batch_size,), dtype=int) * 2

        stimulus1_kwargs = {"theta": theta_1, "modality": modality, "amplitude": amp_1}
        stimulus2_kwargs = {"theta": theta_2, "modality": modality, "amplitude": amp_2}

        context = self.generate_task_period("context", 200, 600)
        stim_1 = self.generate_task_period("stimulus1", 200, 1600, stimulus1_kwargs)
        memory_1 = self.generate_task_period("memory1", 200, 1600)
        stim_2 = self.generate_task_period("stimulus2", 200, 1600, stimulus2_kwargs)
        memory_2 = self.generate_task_period("memory2", 100, 300)
        response = self.generate_task_period("response", 300, 700)

        self.set_task_periods([context, stim_1, memory_1, stim_2, memory_2, response])
        self.set_y(theta_1, theta_2, amp_1, amp_2)

    def set_y(
        self,
        theta_1: torch.Tensor,
        theta_2: torch.Tensor,
        amp_1: torch.Tensor,
        amp_2: torch.Tensor,
    ) -> None:
        """Move to stimulus location with largest amplitude. Only present modality 2."""
        amp = torch.cat((amp_1, amp_2), dim=1)
        amp_idx = torch.argmax(amp, dim=1)

        theta = torch.cat((theta_1, theta_2), dim=1)
        theta = torch.gather(theta, 1, amp_idx.unsqueeze(1))

        response_start = self.n_steps - self.task_periods[-1].n_steps

        self.y[response_start:, :, 1:] = torch.cat(
            (torch.sin(theta), torch.cos(theta)), dim=1
        )
        self.theta[response_start:] = theta.squeeze(1)


@dataclass(repr=False)
class ContextIntModality1(Task):
    batch_size: int = 1
    dt: int = 20
    gamma: float | None = None
    circle: bool = False

    def __post_init__(self) -> None:
        super().__init__("ContextIntModality1", self.batch_size, self.dt, self.gamma)

        # Using values from code
        amp_mean = 0.4 * torch.rand((self.batch_size, 2)) + 0.8
        amp_cohesion_vars = torch.Tensor([0.08, 0.16, 0.32])
        amp_cohesion_sign = torch.Tensor([-1, 1])
        amp_var = (
            amp_cohesion_vars[torch.randint(0, 3, (self.batch_size, 2))]
            * amp_cohesion_sign[torch.randint(0, 2, (self.batch_size, 2))]
        )
        amp_1 = amp_mean + amp_var
        amp_2 = amp_mean - amp_var

        # Generate both modalities at once
        theta_1 = self.generate_stimulus_batch(n_mod=2, circle=self.circle)
        theta_2 = self.generate_stimulus_batch(n_mod=2, circle=self.circle)
        modality = torch.ones((self.batch_size,), dtype=int)

        stimulus1_kwargs = {
            "theta": theta_1,
            "modality": modality,
            "amplitude": amp_1,
            "n_mod": 2,
        }
        stimulus2_kwargs = {
            "theta": theta_2,
            "modality": modality,
            "amplitude": amp_2,
            "n_mod": 2,
        }

        context = self.generate_task_period("context", 200, 600)
        stim_1 = self.generate_task_period("stimulus1", 200, 1600, stimulus1_kwargs)
        memory_1 = self.generate_task_period("memory1", 200, 1600)
        stim_2 = self.generate_task_period("stimulus2", 200, 1600, stimulus2_kwargs)
        memory_2 = self.generate_task_period("memory2", 100, 300)
        response = self.generate_task_period("response", 300, 700)

        self.set_task_periods([context, stim_1, memory_1, stim_2, memory_2, response])
        self.set_y(theta_1, theta_2, amp_1, amp_2)

    def set_y(
        self,
        theta_1: torch.Tensor,
        theta_2: torch.Tensor,
        amp_1: torch.Tensor,
        amp_2: torch.Tensor,
    ) -> None:
        """Filter theta to choose the one with corresponding larger amplitude.
        Both modalities presented, only attend to modality 1."""
        amp_1 = amp_1[:, 0]
        amp_2 = amp_2[:, 0]

        theta_1 = theta_1[:, 0]
        theta_2 = theta_2[:, 0]

        amp = torch.stack((amp_1, amp_2), dim=1)
        amp_idx = torch.argmax(amp, dim=1).unsqueeze(1)

        theta = torch.stack((theta_1, theta_2), dim=1)
        theta = torch.gather(theta, 1, amp_idx)

        response_start = self.n_steps - self.task_periods[-1].n_steps

        self.y[response_start:, :, 1:] = torch.cat(
            (torch.sin(theta), torch.cos(theta)), dim=1
        )
        self.theta[response_start:] = theta.squeeze(1)


@dataclass(repr=False)
class ContextIntModality2(Task):
    batch_size: int = 1
    dt: int = 20
    gamma: float | None = None
    circle: bool = False

    def __post_init__(self) -> None:
        super().__init__("ContextIntModality2", self.batch_size, self.dt, self.gamma)

        # Using values from code
        amp_mean = 0.4 * torch.rand((self.batch_size, 2)) + 0.8
        amp_cohesion_vars = torch.Tensor([0.08, 0.16, 0.32])
        amp_cohesion_sign = torch.Tensor([-1, 1])
        amp_var = (
            amp_cohesion_vars[torch.randint(0, 3, (self.batch_size, 2))]
            * amp_cohesion_sign[torch.randint(0, 2, (self.batch_size, 2))]
        )
        amp_1 = amp_mean + amp_var
        amp_2 = amp_mean - amp_var

        # Generate both modalities at once
        theta_1 = self.generate_stimulus_batch(n_mod=2, circle=self.circle)
        theta_2 = self.generate_stimulus_batch(n_mod=2, circle=self.circle)
        modality = torch.ones((self.batch_size,), dtype=int)

        stimulus1_kwargs = {
            "theta": theta_1,
            "modality": modality,
            "amplitude": amp_1,
            "n_mod": 2,
        }
        stimulus2_kwargs = {
            "theta": theta_2,
            "modality": modality,
            "amplitude": amp_2,
            "n_mod": 2,
        }

        context = self.generate_task_period("context", 200, 600)
        stim_1 = self.generate_task_period("stimulus1", 200, 1600, stimulus1_kwargs)
        memory_1 = self.generate_task_period("memory1", 200, 1600)
        stim_2 = self.generate_task_period("stimulus2", 200, 1600, stimulus2_kwargs)
        memory_2 = self.generate_task_period("memory2", 100, 300)
        response = self.generate_task_period("response", 300, 700)

        self.set_task_periods([context, stim_1, memory_1, stim_2, memory_2, response])
        self.set_y(theta_1, theta_2, amp_1, amp_2)

    def set_y(
        self,
        theta_1: torch.Tensor,
        theta_2: torch.Tensor,
        amp_1: torch.Tensor,
        amp_2: torch.Tensor,
    ) -> None:
        """Attend to stimulus value with largest amplitude.
        Both amplitudes presented, only attend to modality 2."""
        amp_1 = amp_1[:, 1]
        amp_2 = amp_2[:, 1]

        theta_1 = theta_1[:, 1]
        theta_2 = theta_2[:, 1]

        amp = torch.stack((amp_1, amp_2), dim=1)
        amp_idx = torch.argmax(amp, dim=1).unsqueeze(1)

        theta = torch.stack((theta_1, theta_2), dim=1)
        theta = torch.gather(theta, 1, amp_idx)

        response_start = self.n_steps - self.task_periods[-1].n_steps

        self.y[response_start:, :, 1:] = torch.cat(
            (torch.sin(theta), torch.cos(theta)), dim=1
        )
        self.theta[response_start:] = theta.squeeze(1)


@dataclass(repr=False)
class IntegrationMultiModal(Task):
    batch_size: int = 1
    dt: int = 20
    gamma: float | None = None
    circle: bool = False

    def __post_init__(self) -> None:
        super().__init__("IntegrationMultiModal", self.batch_size, self.dt, self.gamma)

        # Using values from paper code
        amp_mean = 0.4 * torch.rand((self.batch_size, 2)) + 0.8
        amp_cohesion_vars = torch.Tensor([0.08, 0.16, 0.32])
        amp_cohesion_sign = torch.Tensor([-1, 1])
        amp_var = (
            amp_cohesion_vars[torch.randint(0, 3, (self.batch_size, 2))]
            * amp_cohesion_sign[torch.randint(0, 2, (self.batch_size, 2))]
        )
        amp_1 = amp_mean + amp_var
        amp_2 = amp_mean - amp_var

        # Generate both modalities at once
        theta_1 = self.generate_stimulus_batch(n_mod=2, circle=self.circle)
        theta_2 = self.generate_stimulus_batch(n_mod=2, circle=self.circle)
        modality = torch.ones((self.batch_size,), dtype=int)

        stimulus1_kwargs = {
            "theta": theta_1,
            "modality": modality,
            "amplitude": amp_1,
            "n_mod": 2,
        }
        stimulus2_kwargs = {
            "theta": theta_2,
            "modality": modality,
            "amplitude": amp_2,
            "n_mod": 2,
        }

        # Define task periods
        context = self.generate_task_period("context", 200, 600)
        stim_1 = self.generate_task_period("stimulus1", 200, 1600, stimulus1_kwargs)
        memory_1 = self.generate_task_period("memory1", 200, 1600)
        stim_2 = self.generate_task_period("stimulus2", 200, 1600, stimulus2_kwargs)
        memory_2 = self.generate_task_period("memory2", 100, 300)
        response = self.generate_task_period("response", 300, 700)

        # Set inputs/outputs
        self.set_task_periods([context, stim_1, memory_1, stim_2, memory_2, response])
        self.set_y(theta_1, theta_2, amp_1, amp_2)

    def set_y(
        self,
        theta_1: torch.Tensor,
        theta_2: torch.Tensor,
        amp_1: torch.Tensor,
        amp_2: torch.Tensor,
    ) -> None:
        """Attend to stimulus value with largest amplitude. Integrate both modalities."""
        amp_1 = amp_1.sum(dim=1)
        amp_2 = amp_2.sum(dim=1)

        theta_1 = theta_1.sum(dim=1)
        theta_2 = theta_2.sum(dim=1)

        # Select the largest amplitude from each stimulus
        amp = torch.stack((amp_1, amp_2), dim=1)
        amp_idx = torch.argmax(amp, dim=1).unsqueeze(1)

        # Grab the stimulus angle indexed by largeset amplitude
        theta = torch.stack((theta_1, theta_2), dim=1)
        theta = torch.gather(theta, 1, amp_idx)

        # Set response period input/output
        response_start = self.n_steps - self.task_periods[-1].n_steps

        self.y[response_start:, :, 1:] = torch.cat(
            (torch.sin(theta), torch.cos(theta)), dim=1
        )
        self.theta[response_start:] = theta.squeeze(1)


@dataclass(repr=False)
class ReactMatch2Sample(Task):
    batch_size: int = 1
    dt: int = 20
    gamma: float | None = None
    circle: bool = False

    def __post_init__(self) -> None:
        super().__init__("ReactMatch2Sample", self.batch_size, self.dt, self.gamma)

        theta_1 = self.generate_stimulus_batch()
        # add minimum [pi/10, 2*pi - pi/10] to avoid falling within acceptable bound
        # Some of these will be set to be the same as the first stimulus randomly
        offset = (2 * torch.pi - torch.pi / 10) * torch.rand(
            (self.batch_size, 1)
        ) + torch.pi / 10
        theta_2 = theta_1 + offset * torch.randint(0, 2, (self.batch_size, 1))

        mod_1 = self.generate_modality(circle=self.circle)
        mod_2 = self.generate_modality(circle=self.circle)

        stimulus1_kwargs = {"theta": theta_1, "modality": mod_1}
        stimulus2_kwargs = {"theta": theta_2, "modality": mod_2}

        context = self.generate_task_period("context", 200, 600)
        stim_1 = self.generate_task_period("stimulus1", 200, 1600, stimulus1_kwargs)
        memory_1 = self.generate_task_period("memory1", 200, 1600)
        stim_2 = self.generate_task_period("stimulus2", 300, 700, stimulus2_kwargs)

        self.set_task_periods([context, stim_1, memory_1, stim_2])
        self.set_y(theta_1, theta_2)

    def set_y(self, theta_1: torch.Tensor, theta_2: torch.Tensor) -> None:
        """Categories match if both stimuli are same value. move to stimulus 2."""
        theta = torch.cat((torch.sin(theta_2), torch.cos(theta_2)), dim=1)
        theta[(theta_1 == theta_2).squeeze(1)] = -1

        response_start = self.n_steps - self.task_periods[-1].n_steps

        self.y[response_start:, :, 1:] = theta
        self.theta[response_start:] = theta_2.squeeze(1)


@dataclass(repr=False)
class ReactNonMatch2Sample(Task):
    batch_size: int = 1
    dt: int = 20
    gamma: Optional[float] = None

    def __post_init__(self) -> None:
        super().__init__("ReactNonMatch2Sample", self.batch_size, self.dt, self.gamma)

        theta_1 = self.generate_stimulus_batch()
        # add minimum [pi/10, 2*pi - pi/10] to avoid falling within acceptable bound
        # Some of these will be set to be the same as the first stimulus randomly
        offset = (2 * torch.pi - torch.pi / 10) * torch.rand(
            (self.batch_size, 1)
        ) + torch.pi / 10
        theta_2 = (
            theta_1 + torch.pi + offset * torch.randint(0, 2, (self.batch_size, 1))
        )

        mod_1 = self.generate_modality()
        mod_2 = self.generate_modality()

        stimulus1_kwargs = {"theta": theta_1, "modality": mod_1}
        stimulus2_kwargs = {"theta": theta_2, "modality": mod_2}

        context = self.generate_task_period("context", 200, 600)
        stim_1 = self.generate_task_period("stimulus1", 200, 1600, stimulus1_kwargs)
        memory_1 = self.generate_task_period("memory1", 200, 1600)
        stim_2 = self.generate_task_period("stimulus2", 300, 700, stimulus2_kwargs)

        self.set_task_periods([context, stim_1, memory_1, stim_2])
        self.set_y(theta_1, theta_2)

    def set_y(self, theta_1: torch.Tensor, theta_2: torch.Tensor) -> None:
        """Categories match if both stimuli differ by pi. move to stimulus 2."""
        # If matched, attend to theta_2
        theta = torch.cat((torch.sin(theta_2), torch.cos(theta_2)), dim=1)
        theta[(theta_2 == theta_1 + torch.pi).squeeze(1)] = -1

        response_start = self.n_steps - self.task_periods[-1].n_steps

        self.y[response_start:, :, 1:] = theta
        self.theta[response_start:] = theta_2.squeeze(1)


@dataclass(repr=False)
class ReactCategoryPro(Task):
    batch_size: int = 1
    dt: int = 20
    gamma: float | None = None
    circle: bool = False

    def __post_init__(self) -> None:
        super().__init__("ReactCategoryPro", self.batch_size, self.dt, self.gamma)

        theta_1 = self.generate_stimulus_batch(circle=self.circle)
        theta_2 = self.generate_stimulus_batch(circle=self.circle)
        mod_1 = self.generate_modality()
        mod_2 = self.generate_modality()

        stimulus1_kwargs = {"theta": theta_1, "modality": mod_1}
        stimulus2_kwargs = {"theta": theta_2, "modality": mod_2}

        context = self.generate_task_period("context", 200, 600)
        stim_1 = self.generate_task_period("stimulus1", 200, 1600, stimulus1_kwargs)
        memory_1 = self.generate_task_period("memory1", 200, 1600)
        stim_2 = self.generate_task_period("stimulus2", 300, 700, stimulus2_kwargs)

        self.set_task_periods([context, stim_1, memory_1, stim_2])
        self.set_y(theta_1, theta_2)

    def set_y(self, theta_1: torch.Tensor, theta_2: torch.Tensor) -> None:
        """Categories match if they are both larger or smaller than pi. move to stimulus 2."""
        cat_1 = torch.logical_and(theta_1 < torch.pi, theta_2 < torch.pi)
        cat_2 = torch.logical_and(theta_1 > torch.pi, theta_2 > torch.pi)

        theta = torch.stack((torch.sin(theta_2), torch.cos(theta_2)), dim=2)
        theta[torch.logical_or(cat_1, cat_2).squeeze(1)] = -1

        response_start = self.n_steps - self.task_periods[-1].n_steps

        self.y[response_start:, :, 1:] = theta.squeeze(1)
        self.theta[response_start:] = theta_2.squeeze(1)


@dataclass(repr=False)
class ReactCategoryAnti(Task):
    batch_size: int = 1
    dt: int = 20
    gamma: float | None = None
    circle: bool = False

    def __post_init__(self) -> None:
        super().__init__("ReactCategoryAnti", self.batch_size, self.dt, self.gamma)

        theta_1 = self.generate_stimulus_batch(circle=self.circle)
        theta_2 = self.generate_stimulus_batch(circle=self.circle)
        mod_1 = self.generate_modality()
        mod_2 = self.generate_modality()

        stimulus1_kwargs = {"theta": theta_1, "modality": mod_1}
        stimulus2_kwargs = {"theta": theta_2, "modality": mod_2}

        context = self.generate_task_period("context", 200, 600)
        stim_1 = self.generate_task_period("stimulus1", 200, 1600, stimulus1_kwargs)
        memory_1 = self.generate_task_period("memory1", 200, 1600)
        stim_2 = self.generate_task_period("stimulus2", 300, 700, stimulus2_kwargs)

        self.set_task_periods([context, stim_1, memory_1, stim_2])
        self.set_y(theta_1, theta_2)

    def set_y(self, theta_1: torch.Tensor, theta_2: torch.Tensor) -> None:
        """Categories match if one is larger and one is smaller than pi. move to stimulus 2."""
        cat_1 = torch.logical_and(theta_1 < torch.pi, theta_2 > torch.pi)
        cat_2 = torch.logical_and(theta_1 > torch.pi, theta_2 < torch.pi)

        theta = torch.cat((torch.sin(theta_2), torch.cos(theta_2)), dim=1)
        theta[torch.logical_or(cat_1, cat_2).squeeze(1)] = -1

        response_start = self.n_steps - self.task_periods[-1].n_steps

        self.y[response_start:, :, 1:] = theta.squeeze(1)
        self.theta[response_start:] = theta_2.squeeze(1)


TASK_DICT = {
    "DelayedPro": DelayedPro,
    "DelayedAnti": DelayedAnti,
    "MemoryPro": MemoryPro,
    "MemoryAnti": MemoryAnti,
    "ReactPro": ReactPro,
    "ReactAnti": ReactAnti,
    "IntegrationModality1": IntegrationModality1,
    "IntegrationModality2": IntegrationModality2,
    "ContextIntModality1": ContextIntModality1,
    "ContextIntModality2": ContextIntModality2,
    "IntegrationMultiModal": IntegrationMultiModal,
    "ReactMatch2Sample": ReactMatch2Sample,
    "ReactNonMatch2Sample": ReactNonMatch2Sample,
    "ReactCategoryPro": ReactCategoryPro,
    "ReactCategoryAnti": ReactCategoryAnti,
}
