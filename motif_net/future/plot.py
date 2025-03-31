import matplotlib.pyplot as plt

import numpy as np
from sklearn.decomposition import PCA


def stim_plotter(theta, z_hat):
    t = np.linspace(0, 2 * np.pi, 500)
    x = np.cos(t)
    y = np.sin(t)

    choice = np.random.randint(0, theta.shape[1])
    theta = theta[-1]
    z_hat = z_hat[-1]

    # sin and cos are flipped
    target_x, target_y = np.cos(theta[choice]), np.sin(theta[choice])

    response_x, response_y = z_hat[choice, 2], z_hat[choice, 1]

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.plot(target_x, target_y, "ro")
    ax.plot(response_x, response_y, "bo")
    return fig, ax


def trial_plotter(z, z_hat, task):
    x = np.arange(z.shape[0])
    choice = np.random.randint(0, z.shape[1])

    fig, axs = plt.subplots(1, 3)
    fig.tight_layout()
    fig.suptitle(f"Task: {task}")

    axs[0].plot(x, z[:, choice, 0], "b-")
    axs[0].plot(x, z_hat[:, choice, 0], "r-")
    axs[0].set_title("Fixation")

    axs[1].plot(x, z[:, choice, 1], "b-")
    axs[1].plot(x, z_hat[:, choice, 1], "r-")
    axs[1].set_title(r"sin $\theta$")

    axs[2].plot(x, z[:, choice, 2], "b-", label="z")
    axs[2].plot(x, z_hat[:, choice, 2], "r-", label="z_hat")
    axs[2].set_title(r"cos $\theta$")
    handles, labels = [], []
    for ax in fig.axes:
        line, label = ax.get_legend_handles_labels()
        handles.extend(line)
        labels.extend(label)
    plt.legend(handles, labels, loc="upper right")

    return fig, axs


def pca_plotter(hidden_state):
    pca = PCA(n_components=2)
    pca.fit(hidden_state)
    lower_dim = pca.transform(hidden_state)
    fig, ax = plt.subplots()
    ax.plot(lower_dim[:, 0], lower_dim[:, 1], "bo-")
    return fig, ax
