import matplotlib.pyplot as plt
import numpy as np


def plot_V(V: np.ndarray, title: str = None) -> plt.figure:
    dealer_showing = np.arange(1, 11, 1)
    player_sum = np.arange(1, 22, 1)

    X, Y = np.meshgrid(dealer_showing, player_sum)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, np.swapaxes(V, 0, 1))
    ax.set_xlabel("Dealer showing")
    ax.set_ylabel("Player sum")
    ax.set_xticks(range(1, 11))
    ax.set_yticks(range(1, 22))
    ax.set_zticks([-1, 0, 1])
    ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Y) / 2))

    if title:
        plt.title(title)

    plt.tight_layout()
    return fig
