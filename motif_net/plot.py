from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from matplotlib.cm import Pastel2

from motif_net import Task, TaskLoader
from experiments.task_init import TASK_DICT

import torch


def _build_output_df(
    theta: np.array, time_spans: dict, label: str, radius: int = 1
) -> pd.DataFrame:
    """Builds a dataframe for plotting purposes.
    Columns are theta (in rad), radius (0 if theta is 0), and label (what type of data)

    Parameters
    ----------
    theta
        A 1D numpy array of the angle at each time step
    time_spans
        Dictionary containing task period names and start and stop times
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
        df.loc[time_spans[period]["start"] : time_spans[period]["end"], "period"] = period
    df["radius"] = radius
    df.loc[df["theta"] == 0, "radius"] = 0
    df["label"] = label

    return df


def _get_positive_theta(sin: np.array, cos: np.array) -> np.array:
    """Takes a set of x and y coordinates and returns positive angles (in rad)

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


def generate_data(
    task: Task, y_hat: Optional[torch.Tensor] = None
) -> tuple[np.array, pd.DataFrame]:
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

    x_fixation = np.expand_dims(x[:, 0, 0], 1)
    x_mod_1 = x[:, 0, 1:3]
    x_mod_2 = x[:, 0, 3:5]
    x_mod_1_theta = _get_positive_theta(x_mod_1[:, 0], x_mod_1[:, 1])
    x_mod_2_theta = _get_positive_theta(x_mod_2[:, 0], x_mod_2[:, 1])

    df_mod1 = _build_output_df(x_mod_1_theta, time_spans, "mod1")
    df_mod2 = _build_output_df(x_mod_2_theta, time_spans, "mod2")

    theta = theta[:, 0].cpu().numpy()
    y = y[:, 0, :].cpu().numpy()
    y_fixation = np.expand_dims(y[:, 0], 1)
    radius = np.sqrt(np.square(y[:, 1]) + np.square(y[:, 2]))
    df_y = _build_output_df(theta, time_spans, "y", radius)

    fixation = np.concat([x_fixation, y_fixation], axis=1)
    df_agg = pd.concat([df_mod1, df_mod2, df_y])

    if y_hat is not None:
        y_hat = y_hat[:, 0, :].detach().cpu().numpy()

        y_hat_fixation = np.expand_dims(y_hat[:, 0, 0], 1)
        fixation = np.concat([fixation, y_hat_fixation], axis=1)
        y_hat_theta = _get_positive_theta(y_hat[:, 1], y_hat[:, 2])
        radius = np.sqrt(np.square(y_hat[:, 1]) + np.square(y_hat[:, 2]))

        df_y_hat = _build_output_df(y_hat_theta, time_spans, "y_hat", radius)
        df_agg = pd.concat([df_agg, df_y_hat])

    return fixation, df_agg


def plot_task(task: Task, y_hat: Optional[torch.Tensor]) -> tuple[plt.Figure, dict]:
    """Plotting function to help evaluate or debug task data or model performance.

    Parameters
    ----------
    task
        Particular task instance to plot
    y_hat
        Model estimation

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
    layout.extend([[f"{label}_{period}" for period in time_spans.keys()] for label in labels])
    projection = {
        key: {"projection": "polar"} for row in layout for key in row if key != "fixation"
    }
    heights = [1.25] + [1] * n_labels

    fig, axd = plt.subplot_mosaic(
        layout, per_subplot_kw=projection, height_ratios=heights, figsize=(24, 16)
    )
    fig.tight_layout()

    # plot fixation
    axd["fixation"].plot(fixation)
    for i, (period, times) in enumerate(time_spans.items()):
        axd["fixation"].axvspan(times["start"], times["end"], 0, 1, color=Pastel2(i), alpha=0.5)
        midpoint = (times["start"] + times["end"]) / 2
        label = f"{period}\nt = {times['end'] - times['start']}"
        axd["fixation"].text(midpoint, 0.05, label, ha="center", va="bottom", size="xx-large")
        axd["fixation"].set_yticks([0, 1])

    xticks = [0] + [time_spans[period]["end"] for period in time_spans]
    axd["fixation"].set_xticks(xticks)
    axd["fixation"].set_xlim(0, time_spans[list(time_spans.keys())[-1]]["end"])

    legend_label = ["input", "expected" "estimate"] if "y_hat" in labels else ["input", "expected"]
    axd["fixation"].legend(legend_label, loc="upper right", fontsize="xx-large")

    # Plot tasks and periods
    for label in labels:
        label_df = data.loc[data["label"] == label]
        for i, period in enumerate(time_spans.keys()):
            period_df = label_df.loc[label_df["period"] == period]
            marker = "o" if label is "y_hat" else "x"
            axd[f"{label}_{period}"].scatter(
                period_df["theta"], period_df["radius"], s=64, marker=marker
            )
            axd[f"{label}_{period}"].set(yticks=[0, 1, 2], yticklabels=[], xticklabels=[])
            axd[f"{label}_{period}"].patch.set(facecolor=Pastel2(i, alpha=0.5))

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

    # axd = {key: ax for key, ax in zip(axd.keys(), axes.flatten())}
    return fig, axd
