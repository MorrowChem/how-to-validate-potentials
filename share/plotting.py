import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def violin(
    x,
    y=None,
    ax=None,
    widths=5,
    force_to_zero=False,
    color="darkseagreen",
    points=100,
):
    if ax is None:
        ax = plt.gca()

    if y is None:
        y = x
        x = np.zeros(len(y))

    data = pd.DataFrame({"x": x, "y": y})

    x_unique = data.x.unique()

    y_by_x = []
    for x_value in x_unique:
        indices = data.x == x_value
        matching_ys = data.y[indices]
        if force_to_zero:
            # add an extra point at 0 to force the violin to start at 0
            matching_ys = [0, *matching_ys]
        y_by_x.append(matching_ys)

    violins = ax.violinplot(
        y_by_x, x_unique, widths=widths, showextrema=False, points=points
    )
    for violin in violins["bodies"]:
        violin.set_facecolor(color)
        violin.set_alpha(0.7)

    if force_to_zero:
        ax.set_ylim(bottom=0)


def rain_cloud(x, y=None, ax=None, strength=5, **kwargs):
    if ax is None:
        ax = plt.gca()
    if y is None:
        y = x
        x = np.zeros(len(y))

    scatter_kwargs = dict(s=1, c="black", linewidth=0)
    scatter_kwargs.update(kwargs)

    noise = np.random.randn(len(x)) * strength
    ax.scatter(x + noise, y, **scatter_kwargs)
