from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from matplotlib.cm import Pastel2
from matplotlib.patches import Rectangle

from .tasks import Task


def _get_positive_theta(sin: npt.NDArray, cos: npt.NDArray) -> npt.NDArray:
    """Takes a set of x and y coordinates and returns positive angles (in rad)

    This is necessary so as non-stimulus blocks have theta set to -1, so the
    plotter assumes theta < 0 is at the fixation spot (0, 0)

    Note
    ----
    Conventially, tuples are ordered (cos, sin). This setup has the opposite in
    all cases: (sin, cos)

    Parameters
    ----------
    sin
        1D numpy array of y values.
    cos
        1D numpy array of x values

    Returns
    -------
    theta
        a 1D numpy array of positive theta angles (in rad). These take values between
        [0, 2 * pi)
    """
    theta = np.atan2(sin, cos) + np.pi * 2
    theta = np.mod(theta, 2 * np.pi)
    return theta


def _build_output_df(
    time_spans: dict,
) -> Callable[[npt.NDArray, str, int], pd.DataFrame]:
    """Closure to build a dataframe where ``time_spans`` is not parameterized.

    Parameters
    time_spans
        Dictionary containing task period names and start and stop times

    Returns
    -------
    build_data_df
        function that creates the dataframe.
    """

    def build_data_df(theta: npt.NDArray, label: str, radius: int = 1) -> pd.DataFrame:
        """Builds a dataframe for plotting purposes.
        Columns are theta (in rad), radius (0 if theta is 0), and label (what type of data)

        Parameters
        ----------
        theta
            A 1D numpy array of the angle at each time step
        label
            The label to designate the data type (input, output, estimation, etc.)
        radius
            The radius that will be plotted. By default this is 1, but will go to 0
            if theta is 0. This is due to the low probability that the 0 angle is actually
            chosen.

        Returns
        -------
        df
            A dataframe formatted for the task/output plotting function
        """
        periods = list(time_spans.keys())
        df = pd.DataFrame({"theta": np.max((theta, np.zeros(theta.shape)), axis=0)})
        for period in periods:
            df.loc[
                time_spans[period]["start"] : time_spans[period]["end"], "period"
            ] = period
        df["radius"] = radius
        df.loc[df["theta"] == 0, "radius"] = 0
        df["label"] = label

        return df

    return build_data_df


def _get_radius(x1: npt.NDArray, x2) -> npt.NDArray:
    """Gets the radius from the xy coordinates of points

    Parameters
    ----------
    x1
        np array of either x or y coords
    x2
        np array of either y or x coords

    Returns
    -------
    radius
        A 1D np array of radii
    """
    radius = np.sqrt(np.square(x1) + np.square(x2))
    return radius


def generate_data(
    task: Task, y_hat: Optional[torch.Tensor] = None
) -> tuple[npt.NDArray, pd.DataFrame]:
    """Takes a task and optionally a model estimate and plots for review.

    Note that this takes the first index for batched data.

    Parameters
    ----------
    task
        The particular task instantiation to plot
    y_hat
        The model estimation output

    Returns
    -------
    fixation
        A 1D or 2D numpy array of fixation values for each timestep
    df_agg
        A plotting friendly dataframe
    """
    time_spans = task.get_time_spans()
    x = task.get_input_array().cpu().numpy()
    y, theta = task.get_output()
    build_data_df = _build_output_df(time_spans)

    x_fixation = np.expand_dims(x[:, 0, 0], 1)
    x_mod1 = x[:, 0, 1:3]
    x_mod2 = x[:, 0, 3:5]
    x_mod1_theta = _get_positive_theta(x_mod1[:, 0], x_mod1[:, 1])
    x_mod2_theta = _get_positive_theta(x_mod2[:, 0], x_mod2[:, 1])

    x_mod1_radius = _get_radius(x_mod1[:, 0], x_mod1[:, 1])
    x_mod2_radius = _get_radius(x_mod2[:, 0], x_mod2[:, 1])

    df_mod1 = build_data_df(x_mod1_theta, "mod1", x_mod1_radius)
    df_mod2 = build_data_df(x_mod2_theta, "mod2", x_mod2_radius)

    theta = theta[:, 0].cpu().numpy()
    y = y[:, 0, :].cpu().numpy()
    y_fixation = np.expand_dims(y[:, 0], 1)
    radius = _get_radius(y[:, 1], y[:, 2])
    df_y = build_data_df(theta, "y", radius)

    fixation = np.concat([x_fixation, y_fixation], axis=1)
    df_agg = pd.concat([df_mod1, df_mod2, df_y])

    if y_hat is not None:
        y_hat = y_hat[:, 0, :].detach().cpu().numpy()

        y_hat_fixation = np.expand_dims(y_hat[:, 0], 1)
        fixation = np.concat([fixation, y_hat_fixation], axis=1)
        y_hat_theta = _get_positive_theta(y_hat[:, 1], y_hat[:, 2])
        radius = _get_radius(y_hat[:, 1], y_hat[:, 2])

        df_y_hat = build_data_df(y_hat_theta, "y_hat", radius)
        df_agg = pd.concat([df_agg, df_y_hat])

    return fixation, df_agg


def plot_task(
    task: Task, y_hat: Optional[torch.Tensor], iterations: int | None = None
) -> tuple[plt.Figure, dict]:
    """Plotting function to help evaluate or debug task data or model performance.

    Parameters
    ----------
    task
        Particular task instance to plot
    y_hat
        Model estimation
    iterations
        The number of iterations the plot was generated after. Used for plot subtitle

    Returns
    -------
    fig
        The matplotlib figure
    axd
        A dictionary of the ax subplots
    """
    time_spans = task.get_time_spans()
    fixation, data = generate_data(task, y_hat)

    n_tasks = len(time_spans)
    labels = data["label"].unique()
    n_labels = labels.shape[0]

    layout = [["fixation"] * n_tasks]
    layout.extend(
        [[f"{label}_{period}" for period in time_spans.keys()] for label in labels]
    )
    projection = {
        key: {"projection": "polar"}
        for row in layout
        for key in row
        if key != "fixation"
    }
    heights = [1.25] + [1] * n_labels

    fig, axd = plt.subplot_mosaic(
        layout, per_subplot_kw=projection, height_ratios=heights, figsize=(24, 16)
    )
    fig.tight_layout()
    title = task.name + (
        f" - Iterations: {iterations:,}" if iterations is not None else ""
    )
    fig.suptitle(title)
    cmap = Pastel2

    # plot fixation
    axd["fixation"].plot(fixation)
    for i, (period, times) in enumerate(time_spans.items()):
        axd["fixation"].axvspan(
            times["start"], times["end"], 0, 1, color=cmap(i), alpha=0.5
        )
        midpoint = (times["start"] + times["end"]) / 2
        label = f"{period}\nt = {times['end'] - times['start']}"
        axd["fixation"].text(
            midpoint, 0.05, label, ha="center", va="bottom", size="xx-large"
        )
        axd["fixation"].set_yticks([0, 1])

    xticks = [0] + [time_spans[period]["end"] for period in time_spans]
    axd["fixation"].set_xticks(xticks)
    axd["fixation"].set_xlim(0, time_spans[list(time_spans.keys())[-1]]["end"])

    legend_label = (
        ["input", "expected", "estimate"]
        if "y_hat" in labels
        else ["input", "expected"]
    )
    axd["fixation"].legend(legend_label, loc="upper right", fontsize="xx-large")

    # Plot tasks and periods
    color = {
        "mod1": "tab:blue",
        "mod2": "tab:blue",
        "y": "tab:orange",
        "y_hat": "tab:green",
    }
    for label in labels:
        label_df = data.loc[data["label"] == label]

        for i, period in enumerate(time_spans.keys()):
            period_df = label_df.loc[label_df["period"] == period]
            marker = "o" if label == "y_hat" else "x"
            axd[f"{label}_{period}"].scatter(
                period_df["theta"],
                period_df["radius"],
                s=64,
                marker=marker,
                c=color[label],
            )
            axd[f"{label}_{period}"].set(
                yticks=[0, 1, 2], yticklabels=[], xticklabels=[]
            )
            axd[f"{label}_{period}"].patch.set(facecolor=cmap(i, alpha=0.5))

        # Add seperation for each data type
        bbox = axd[f"{label}_context"].get_position(original=True)
        rect = Rectangle(
            (bbox.x0, bbox.y0 - 0.01), 0.97, bbox.height + 0.02, zorder=-1, fill=False
        )
        fig.add_artist(rect)
        bbox = rect.get_bbox()
        rect_midpoint = bbox.y0 + bbox.height / 2
        text = plt.Text(
            bbox.x0 - 0.005,
            rect_midpoint,
            label,
            ha="right",
            va="bottom",
            rotation=90,
            size="xx-large",
        )
        fig.add_artist(text)

    return fig, axd


def plot_performance(loss: torch.Tensor, iteration: int) -> tuple[plt.Figure, plt.Axes]:
    """Plots the running loss of the model over time.

    Parameters
    ----------
    loss
        A 1D tensor object with the model's loss
    iteration
        The current iteration step the model is in. Used for the title.

    Returns
    -------
    fig
        The ``matplotlib`` figure object
    ax
        The ``matplotlib`` axis object
    """
    loss = loss.cpu().numpy()

    fig, ax = plt.subplots(1, 1)
    fig.tight_layout()

    time_step = 5000
    ticks = np.arange(0, iteration + 1, time_step)
    ax.semilogy(ticks, loss)
    ax.set_xticks(ticks)
    ax.set_title(f"Iteration {iteration:,} - Current loss: {loss[-1]}")
    return fig, ax
