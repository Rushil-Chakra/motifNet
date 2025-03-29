from typing import Union, Optional
from .tasks import Task, TaskPeriod

import torch
import numpy as np


class DelayedPro(Task):
    def __init__(
        self,
        batch_size: int = 64,
        dt: int = 20,
        gamma: Optional[float] = None,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__("DelayedPro", batch_size, dt, gamma, rng)

        theta = self.generate_stimulus_batch()
        modality = self.generate_modality()
        context = TaskPeriod(
            "context",
            self.name,
            min_t=300,
            max_t=700,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1 = TaskPeriod(
            "stimulus1",
            self.name,
            min_t=300,
            max_t=1500,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1.add_stimulus_values(theta, modality)
        response = TaskPeriod(
            "response",
            self.name,
            min_t=300,
            max_t=1500,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        response.add_stimulus_values(theta, modality)
        self.set_task_periods([context, stim_1, response])
        self.set_z(theta)

    def set_z(self, theta: torch.Tensor) -> None:
        """Move to same angle as presented."""
        response_start = self.n_steps - self.task_periods[-1].n_steps
        self.z[response_start:, :, 1:] = torch.cat(
            (torch.sin(theta), torch.cos(theta)), dim=1
        )
        self.theta[response_start:] = theta.squeeze(1)


class DelayedAnti(Task):
    def __init__(
        self,
        batch_size: int = 64,
        dt: int = 20,
        gamma: Optional[float] = None,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__("DelayedAnti", batch_size, dt, gamma, rng)

        theta = self.generate_stimulus_batch()
        modality = self.generate_modality()
        context = TaskPeriod(
            "context",
            self.name,
            min_t=300,
            max_t=700,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1 = TaskPeriod(
            "stimulus1",
            self.name,
            min_t=300,
            max_t=1500,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1.add_stimulus_values(theta, modality)
        response = TaskPeriod(
            "response",
            self.name,
            min_t=300,
            max_t=1500,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        response.add_stimulus_values(theta, modality)
        self.set_task_periods([context, stim_1, response])
        self.set_z(theta)

    def set_z(self, theta: torch.Tensor) -> None:
        """Move in opposite direction of angle presented."""
        theta += torch.pi
        response_start = self.n_steps - self.task_periods[-1].n_steps
        self.z[response_start:, :, 1:] = torch.cat(
            (torch.sin(theta), torch.cos(theta)), dim=1
        )
        self.theta[response_start:] = theta.squeeze(1)


class MemoryPro(Task):
    def __init__(
        self,
        batch_size: int = 64,
        dt: int = 20,
        gamma: Optional[float] = None,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__("MemoryPro", batch_size, dt, gamma, rng)

        theta = self.generate_stimulus_batch()
        modality = self.generate_modality()
        context = TaskPeriod(
            "context",
            self.name,
            min_t=300,
            max_t=700,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1 = TaskPeriod(
            "stimulus1",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1.add_stimulus_values(theta, modality)
        memory_1 = TaskPeriod(
            "memory1",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        response = TaskPeriod(
            "response",
            self.name,
            min_t=300,
            max_t=700,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        self.set_task_periods([context, stim_1, memory_1, response])
        self.set_z(theta)

    def set_z(self, theta: torch.Tensor) -> None:
        """Move in same angle as presented after stimulus has faded."""
        response_start = self.n_steps - self.task_periods[-1].n_steps
        self.z[response_start:, :, 1:] = torch.cat(
            (torch.sin(theta), torch.cos(theta)), dim=1
        )
        self.theta[response_start:] = theta.squeeze(1)


class MemoryAnti(Task):
    def __init__(
        self,
        batch_size: int = 64,
        dt: int = 20,
        gamma: Optional[float] = None,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__("MemoryAnti", batch_size, dt, gamma, rng)

        theta = self.generate_stimulus_batch()
        modality = self.generate_modality()
        context = TaskPeriod(
            "context",
            self.name,
            min_t=300,
            max_t=700,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1 = TaskPeriod(
            "stimulus1",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1.add_stimulus_values(theta, modality)
        memory_1 = TaskPeriod(
            "memory1",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        response = TaskPeriod(
            "response",
            self.name,
            min_t=300,
            max_t=700,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        self.set_task_periods([context, stim_1, memory_1, response])
        self.set_z(theta)

    def set_z(self, theta: torch.Tensor) -> None:
        """Move in opposite direction of stimulus after it has faded."""
        theta += torch.pi
        response_start = self.n_steps - self.task_periods[-1].n_steps
        self.z[response_start:, :, 1:] = torch.cat(
            (torch.sin(theta), torch.cos(theta)), dim=1
        )
        self.theta[response_start:] = theta.squeeze(1)


class ReactPro(Task):
    def __init__(
        self,
        batch_size: int = 64,
        dt: int = 20,
        gamma: Optional[float] = None,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__("ReactPro", batch_size, dt, gamma, rng)

        theta = self.generate_stimulus_batch()
        modality = self.generate_modality()

        context = TaskPeriod(
            "context",
            self.name,
            min_t=500,
            max_t=2500,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        # Response period as well, but react tasks have fixation = 1 even during response
        stim_1 = TaskPeriod(
            "stimulus1",
            self.name,
            min_t=300,
            max_t=1700,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1.add_stimulus_values(theta, modality)
        self.set_task_periods([context, stim_1])
        self.set_z(theta)

    def set_z(self, theta: torch.Tensor) -> None:
        """Move in direction of stimulus immediately. There is no change in fixation during
        the whole trial."""
        response_start = self.n_steps - self.task_periods[-1].n_steps
        self.z[response_start:, :, 1:] = torch.cat(
            (torch.sin(theta), torch.cos(theta)), dim=1
        )
        self.theta[response_start:] = theta.squeeze(1)


class ReactAnti(Task):
    def __init__(
        self,
        batch_size: int = 64,
        dt: int = 20,
        gamma: Optional[float] = None,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__("ReactAnti", batch_size, dt, gamma, rng)

        theta = self.generate_stimulus_batch()
        modality = self.generate_modality()

        context = TaskPeriod(
            "context",
            self.name,
            min_t=500,
            max_t=2500,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        # Response period as well, but react tasks have fixation = 1 even during response
        stim_1 = TaskPeriod(
            "stimulus1",
            self.name,
            min_t=300,
            max_t=1700,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1.add_stimulus_values(theta, modality)
        self.set_task_periods([context, stim_1])
        self.set_z(theta)

    def set_z(self, theta):
        """Move in direction of stimulus immediately. There is no change in fixation during
        the whole trial."""
        theta += torch.pi
        response_start = self.n_steps - self.task_periods[-1].n_steps
        self.z[response_start:, :, 1:] = torch.cat(
            (torch.sin(theta), torch.cos(theta)), dim=1
        )
        self.theta[response_start:] = theta.squeeze(1)


class IntegrationModality1(Task):
    def __init__(
        self,
        batch_size: int = 64,
        dt: int = 20,
        gamma: Optional[float] = None,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__("IntegrationModality1", batch_size, dt, gamma, rng)

        # Using values from code
        amp_mean = 0.4 * torch.rand((batch_size, 1)) + 0.8
        amp_var = 1.6 * torch.rand((batch_size, 1)) - 0.8
        amp_1 = amp_mean + amp_var
        amp_2 = amp_mean - amp_var
        theta_1 = self.generate_stimulus_batch()
        theta_2 = self.generate_stimulus_batch()

        modality = torch.ones((batch_size,), dtype=int)

        context = TaskPeriod(
            "context",
            self.name,
            min_t=200,
            max_t=600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1 = TaskPeriod(
            "stimulus1",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1.add_stimulus_values(theta_1, modality, amp_1)
        memory_1 = TaskPeriod(
            "memory1",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_2 = TaskPeriod(
            "stimulus2",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_2.add_stimulus_values(theta_2, modality, amp_2)
        memory_2 = TaskPeriod(
            "memory2",
            self.name,
            min_t=100,
            max_t=300,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        response = TaskPeriod(
            "response",
            self.name,
            min_t=300,
            max_t=700,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        self.set_task_periods([context, stim_1, memory_1, stim_2, memory_2, response])
        self.set_z(theta_1, theta_2, amp_1, amp_2)

    def set_z(
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
        self.z[response_start:, :, 1:] = torch.cat(
            (torch.sin(theta), torch.cos(theta)), dim=1
        )
        self.theta[response_start, :] = theta.squeeze(1)


class IntegrationModality2(Task):
    def __init__(
        self,
        batch_size: int = 64,
        dt: int = 20,
        gamma: Optional[float] = None,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__("IntegrationModality2", batch_size, dt, gamma, rng)

        # Using values from code
        amp_mean = 0.4 * torch.rand((batch_size, 1)) + 0.8
        amp_var = 1.6 * torch.rand((batch_size, 1)) - 0.8
        amp_1 = amp_mean + amp_var
        amp_2 = amp_mean - amp_var
        theta_1 = self.generate_stimulus_batch()
        theta_2 = self.generate_stimulus_batch()

        modality = torch.ones((batch_size,), dtype=int) * 2

        context = TaskPeriod(
            "context",
            self.name,
            min_t=200,
            max_t=600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1 = TaskPeriod(
            "stimulus1",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1.add_stimulus_values(theta_1, modality, amp_1)
        memory_1 = TaskPeriod(
            "memory1",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_2 = TaskPeriod(
            "stimulus2",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_2.add_stimulus_values(theta_2, modality, amp_2)
        memory_2 = TaskPeriod(
            "memory2",
            self.name,
            min_t=100,
            max_t=300,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        response = TaskPeriod(
            "response",
            self.name,
            min_t=300,
            max_t=700,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        self.set_task_periods([context, stim_1, memory_1, stim_2, memory_2, response])
        self.set_z(theta_1, theta_2, amp_1, amp_2)

    def set_z(
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
        self.z[response_start:, :, 1:] = torch.cat(
            (torch.sin(theta), torch.cos(theta)), dim=1
        )
        self.theta[response_start:] = theta.squeeze(1)


class ContextIntModality1(Task):
    def __init__(
        self,
        batch_size: int = 64,
        dt: int = 20,
        gamma: Optional[float] = None,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__("ContextIntModality1", batch_size, dt, gamma, rng)

        # Using values from code
        # matrix math means I can set both modalities at once
        amp_mean = 0.4 * torch.rand((batch_size, 2)) + 0.8
        amp_var = 1.6 * torch.rand((batch_size, 2)) - 0.8
        amp_1 = amp_mean + amp_var
        amp_2 = amp_mean - amp_var

        # Generate both modalities at once
        theta_1 = self.generate_stimulus_batch(n_mod=2)
        theta_2 = self.generate_stimulus_batch(n_mod=2)
        modality = torch.ones((batch_size,), dtype=int)
        context = TaskPeriod(
            "context",
            self.name,
            min_t=200,
            max_t=600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1 = TaskPeriod(
            "stimulus1",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1.add_stimulus_values(theta_1, modality, amp_1, n_mod=2)
        memory_1 = TaskPeriod(
            "memory1",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_2 = TaskPeriod(
            "stimulus2",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_2.add_stimulus_values(theta_2, modality, amp_2, n_mod=2)
        memory_2 = TaskPeriod(
            "memory2",
            self.name,
            min_t=100,
            max_t=300,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        response = TaskPeriod(
            "response",
            self.name,
            min_t=300,
            max_t=700,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        self.set_task_periods([context, stim_1, memory_1, stim_2, memory_2, response])
        self.set_z(theta_1, theta_2, amp_1, amp_2)

    def set_z(
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
        amp_idx = torch.argmax(amp, dim=1)
        theta = torch.stack((theta_1, theta_2), dim=1)
        theta = torch.gather(theta, 1, amp_idx.unsqueeze(1))
        response_start = self.n_steps - self.task_periods[-1].n_steps
        self.z[response_start:, :, 1:] = torch.cat(
            (torch.sin(theta), torch.cos(theta)), dim=1
        )
        self.theta[response_start:] = theta.squeeze(1)


class ContextIntModality2(Task):
    def __init__(
        self,
        batch_size: int = 64,
        dt: int = 20,
        gamma: Optional[float] = None,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__("ContextIntModality2", batch_size, dt, gamma, rng)

        # Using values from code
        amp_mean = 0.4 * torch.rand((batch_size, 2)) + 0.8
        amp_var = 1.6 * torch.rand((batch_size, 2)) - 0.8
        amp_1 = amp_mean + amp_var
        amp_2 = amp_mean - amp_var

        # Generate both modalities at once
        theta_1 = self.generate_stimulus_batch(n_mod=2)
        theta_2 = self.generate_stimulus_batch(n_mod=2)
        modality = torch.ones((batch_size,), dtype=int)

        context = TaskPeriod(
            "context",
            self.name,
            min_t=200,
            max_t=600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1 = TaskPeriod(
            "stimulus1",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1.add_stimulus_values(theta_1, modality, amp_1, n_mod=2)
        memory_1 = TaskPeriod(
            "memory1",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_2 = TaskPeriod(
            "stimulus2",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_2.add_stimulus_values(theta_2, modality, amp_2, n_mod=2)
        memory_2 = TaskPeriod(
            "memory2",
            self.name,
            min_t=100,
            max_t=300,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        response = TaskPeriod(
            "response",
            self.name,
            min_t=300,
            max_t=700,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        self.set_task_periods([context, stim_1, memory_1, stim_2, memory_2, response])
        self.set_z(theta_1, theta_2, amp_1, amp_2)

    def set_z(
        self,
        theta_1: torch.Tensor,
        theta_2: torch.Tensor,
        amp_1: torch.Tensor,
        amp_2: torch.Tensor,
    ) -> None:
        """Attend to stimulus value with largest amplitude.
        Both amplitudes presented, only attend to modality 2."""
        amp_1 = amp_1[:, 1].unsqueeze(1)
        amp_2 = amp_2[:, 1].unsqueeze(1)
        theta_1 = theta_1[:, 1].unsqueeze(1)
        theta_2 = theta_2[:, 1].unsqueeze(1)

        amp = torch.cat((amp_1, amp_2), dim=1)
        amp_idx = torch.argmax(amp, dim=1)
        theta = torch.cat((theta_1, theta_2), dim=1)
        theta = torch.gather(theta, 1, amp_idx.unsqueeze(1))
        response_start = self.n_steps - self.task_periods[-1].n_steps
        self.z[response_start:, :, 1:] = torch.cat(
            (torch.sin(theta), torch.cos(theta)), dim=1
        )
        self.theta[response_start:] = theta.squeeze(1)


class IntegrationMultiModal(Task):
    def __init__(
        self,
        batch_size: int = 64,
        dt: int = 20,
        gamma: Optional[float] = None,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__("IntegrationMultiModal", batch_size, dt, gamma, rng)

        # Using values from code
        amp_mean = 0.4 * torch.rand((batch_size, 2)) + 0.8
        amp_var = 1.6 * torch.rand((batch_size, 2)) - 0.8
        amp_1 = amp_mean + amp_var
        amp_2 = amp_mean - amp_var

        # Generate both modalities at once
        theta_1 = self.generate_stimulus_batch(n_mod=2)
        theta_2 = self.generate_stimulus_batch(n_mod=2)
        modality = torch.ones((batch_size,), dtype=int)

        context = TaskPeriod(
            "context",
            self.name,
            min_t=200,
            max_t=600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1 = TaskPeriod(
            "stimulus1",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1.add_stimulus_values(theta_1, modality, amp_1, n_mod=2)
        memory_1 = TaskPeriod(
            "memory1",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_2 = TaskPeriod(
            "stimulus2",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_2.add_stimulus_values(theta_2, modality, amp_2, n_mod=2)
        memory_2 = TaskPeriod(
            "memory2",
            self.name,
            min_t=100,
            max_t=300,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        response = TaskPeriod(
            "response",
            self.name,
            min_t=300,
            max_t=700,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        self.set_task_periods([context, stim_1, memory_1, stim_2, memory_2, response])
        self.set_z(theta_1, theta_2, amp_1, amp_2)

    def set_z(
        self,
        theta_1: torch.Tensor,
        theta_2: torch.Tensor,
        amp_1: torch.Tensor,
        amp_2: torch.Tensor,
    ) -> None:
        """Attend to stimulus value with largest amplitude. Integrate both modalities."""
        amp_1 = amp_1.sum(dim=1).unsqueeze(1)
        amp_2 = amp_2.sum(dim=1).unsqueeze(1)
        theta_1 = theta_1.sum(dim=1).unsqueeze(1)
        theta_2 = theta_2.sum(dim=1).unsqueeze(1)

        amp = torch.cat((amp_1, amp_2), dim=1)
        amp_idx = torch.argmax(amp, dim=1)
        theta = torch.cat((theta_1, theta_2), dim=1)
        theta = torch.gather(theta, 1, amp_idx.unsqueeze(1))
        response_start = self.n_steps - self.task_periods[-1].n_steps
        self.z[response_start:, :, 1:] = torch.cat(
            (torch.sin(theta), torch.cos(theta)), dim=1
        )
        self.theta[response_start:] = theta.squeeze(1)


class ReactMatch2Sample(Task):
    def __init__(
        self,
        batch_size: int = 64,
        dt: int = 20,
        gamma: Optional[float] = None,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__("ReactMatch2Sample", batch_size, dt, gamma, rng)

        theta_1 = self.generate_stimulus_batch()
        # add minimum [pi/10, 2*pi - pi/10] to avoid falling within acceptable bound
        # Some of these will be set to be the same as the first stimulus randomly
        offset = (2 * torch.pi - torch.pi / 5) * torch.rand(
            (batch_size, 1)
        ) + torch.pi / 10
        theta_2 = theta_1 + offset * torch.randint(0, 2, (batch_size, 1))

        mod_1 = self.generate_modality()
        mod_2 = self.generate_modality()

        context = TaskPeriod(
            "context",
            self.name,
            min_t=200,
            max_t=600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1 = TaskPeriod(
            "stimulus1",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1.add_stimulus_values(theta_1, mod_1)
        memory_1 = TaskPeriod(
            "memory1",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_2 = TaskPeriod(
            "stimulus2",
            self.name,
            min_t=300,
            max_t=700,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_2.add_stimulus_values(theta_2, mod_2)
        self.set_task_periods([context, stim_1, memory_1, stim_2])
        self.set_z(theta_1, theta_2)

    def set_z(self, theta_1: torch.Tensor, theta_2: torch.Tensor) -> None:
        """Categories match if both stimuli are same value. move to stimulus 2."""
        theta = torch.cat((torch.sin(theta_2), torch.cos(theta_2)), dim=1)
        theta[(theta_1 == theta_2).squeeze(1)] = -1
        response_start = self.n_steps - self.task_periods[-1].n_steps
        self.z[response_start:, :, 1:] = theta
        self.theta[response_start:] = theta_2.squeeze(1)


class ReactNonMatch2Sample(Task):
    def __init__(
        self,
        batch_size: int = 64,
        dt: int = 20,
        gamma: Optional[float] = None,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__("ReactNonMatch2Sample", batch_size, dt, gamma, rng)

        theta_1 = self.generate_stimulus_batch()
        # add minimum [pi/10, 2*pi - pi/10] to avoid falling within acceptable bound
        # Some of these will be set to be the same as the first stimulus randomly
        offset = (2 * torch.pi - torch.pi / 10) * torch.rand(
            (batch_size, 1)
        ) + torch.pi / 10
        theta_2 = theta_1 + torch.pi + offset * torch.randint(0, 2, (batch_size, 1))

        mod_1 = self.generate_modality()
        mod_2 = self.generate_modality()

        context = TaskPeriod(
            "context",
            self.name,
            min_t=200,
            max_t=600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1 = TaskPeriod(
            "stimulus1",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1.add_stimulus_values(theta_1, mod_1)
        memory_1 = TaskPeriod(
            "memory1",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_2 = TaskPeriod(
            "stimulus2",
            self.name,
            min_t=300,
            max_t=700,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_2.add_stimulus_values(theta_2, mod_2)
        self.set_task_periods([context, stim_1, memory_1, stim_2])
        self.set_z(theta_1, theta_2)

    def set_z(self, theta_1: torch.Tensor, theta_2: torch.Tensor) -> None:
        """Categories match if both stimuli differ by pi. move to stimulus 2."""
        # If matched, attend to theta_2
        theta = torch.cat((torch.sin(theta_2), torch.cos(theta_2)), dim=1)
        theta[(theta_2 == theta_1 + torch.pi).squeeze(1)] = -1
        response_start = self.n_steps - self.task_periods[-1].n_steps
        self.z[response_start:, :, 1:] = theta
        self.theta[response_start:] = theta_2.squeeze(1)


class ReactCategoryPro(Task):
    def __init__(
        self,
        batch_size: int = 64,
        dt: int = 20,
        gamma: Optional[float] = None,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__("ReactCategoryPro", batch_size, dt, gamma, rng)

        theta_1 = self.generate_stimulus_batch()
        theta_2 = self.generate_stimulus_batch()
        mod_1 = self.generate_modality()
        mod_2 = self.generate_modality()

        context = TaskPeriod(
            "context",
            self.name,
            min_t=200,
            max_t=600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1 = TaskPeriod(
            "stimulus1",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1.add_stimulus_values(theta_1, mod_1)
        memory_1 = TaskPeriod(
            "memory1",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_2 = TaskPeriod(
            "stimulus2",
            self.name,
            min_t=300,
            max_t=700,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_2.add_stimulus_values(theta_2, mod_2)
        self.set_task_periods([context, stim_1, memory_1, stim_2])
        self.set_z(theta_1, theta_2)

    def set_z(self, theta_1: torch.Tensor, theta_2: torch.Tensor) -> None:
        """Categories match if they are both larger or smaller than pi. move to stimulus 2."""
        cat_1 = torch.logical_and(theta_1 < torch.pi, theta_2 < torch.pi)
        cat_2 = torch.logical_and(theta_1 > torch.pi, theta_2 > torch.pi)

        theta = torch.stack((torch.sin(theta_2), torch.cos(theta_2)), dim=2)
        theta[torch.logical_or(cat_1, cat_2).squeeze(1)] = -1
        response_start = self.n_steps - self.task_periods[-1].n_steps
        self.z[response_start:, :, 1:] = theta.squeeze(1)
        self.theta[response_start:] = theta_2.squeeze(1)


class ReactCategoryAnti(Task):
    def __init__(
        self,
        batch_size: int = 64,
        dt: int = 20,
        gamma: Optional[float] = None,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__("ReactCategoryAnti", batch_size, dt, gamma, rng)

        theta_1 = self.generate_stimulus_batch()
        theta_2 = self.generate_stimulus_batch()
        mod_1 = self.generate_modality()
        mod_2 = self.generate_modality()

        context = TaskPeriod(
            "context",
            self.name,
            min_t=200,
            max_t=600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1 = TaskPeriod(
            "stimulus1",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_1.add_stimulus_values(theta_1, mod_1)
        memory_1 = TaskPeriod(
            "memory1",
            self.name,
            min_t=200,
            max_t=1600,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_2 = TaskPeriod(
            "stimulus2",
            self.name,
            min_t=300,
            max_t=700,
            dt=self.dt,
            batch_size=self.batch_size,
            rng=self.rng,
        )
        stim_2.add_stimulus_values(theta_2, mod_2)
        self.set_task_periods([context, stim_1, memory_1, stim_2])
        self.set_z(theta_1, theta_2)

    def set_z(self, theta_1: torch.Tensor, theta_2: torch.Tensor) -> None:
        """Categories match if one is larger and one is smaller than pi. move to stimulus 2."""
        cat_1 = torch.logical_and(theta_1 < torch.pi, theta_2 > torch.pi)
        cat_2 = torch.logical_and(theta_1 > torch.pi, theta_2 < torch.pi)

        theta = torch.cat((torch.sin(theta_2), torch.cos(theta_2)), dim=1)
        theta[torch.logical_or(cat_1, cat_2).squeeze(1)] = -1
        response_start = self.n_steps - self.task_periods[-1].n_steps
        self.z[response_start:, :, 1:] = theta.squeeze(1)
        self.theta[response_start:] = theta_2.squeeze(1)


def task_generator(task_list: Optional[list[Task]] = None, kwargs: dict = {}) -> Task:
    """Randomly picks a task.

    Parameters
    ----------
    task_list:
        Custom list of tasks to select from. By default all tasks are considered
        with ``ContextIntModality1`` and ``ContextIntModality2`` are each 5x more likely to be chosen.
    kwargs
        Named parameters to pass to task initialization.

    Returns
    -------
    task
        A randomly chosen task.
    """
    # This is horrible and seriously needs to be refactrored
    if task_list is None:
        task_list = [
            "DelayedPro",
            "DelayedAnti",
            "MemoryPro",
            "MemoryAnti",
            "ReactPro",
            "ReactAnti",
            "IntegrationModality1",
            "IntegrationModality2",
            "ContextIntModality1",
            "ContextIntModality1",
            "ContextIntModality1",
            "ContextIntModality1",
            "ContextIntModality1",
            "ContextIntModality2",
            "ContextIntModality2",
            "ContextIntModality2",
            "ContextIntModality2",
            "ContextIntModality2",
            "ContextIntModality2",
            "IntegrationMultiModal",
            "ReactMatch2Sample",
            "ReactNonMatch2Sample",
            "ReactCategoryPro",
            "ReactCategoryAnti",
        ]
    task_dict = {
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
    task_object = task_dict[task_list[np.random.randint(0, len(task_list))]]
    task = task_object(**kwargs)
    return task
