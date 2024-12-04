import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score


def plot_scatter(
    preds: dict,
    z_test: np.ndarray,
    data_lower_lim: float = 0.0,
    data_upper_lim: float = 0.6,
    save_loc: str = "scatter.png",
) -> None:
    """Functionality to plot redshift scatter plots for different models."""
    fig, ax = plt.subplots(2, len(preds.keys()), figsize=(16, 10))

    for i, name in enumerate(preds.keys()):
        sns.scatterplot(ax=ax[0], x=z_test, y=preds[name], s=5, color=".15")
        sns.histplot(
            ax=ax[0], x=z_test, y=preds[name], bins=50, pthresh=0.1, cmap="mako"
        )
        sns.kdeplot(
            ax=ax[0], x=z_test, y=preds[name], levels=5, color="k", linewidths=1
        )

        ax[0].plot(
            data_lower_lim,
            data_upper_lim * 1.1,
            "--",
            linewidth=1.5,
            alpha=0.5,
            color="grey",
        )
        ax[0].set_xlim(data_lower_lim, data_upper_lim)
        ax[0].set_ylim(data_lower_lim, data_upper_lim)
        ax[0].text(
            0.9,
            0.1,
            "$R^2$ score: %0.2f" % r2_score(z_test, preds[name]),
            horizontalalignment="right",
            verticalalignment="top",
            fontsize=22,
            transform=ax[0].transAxes,
        )
        ax[0].set_title(name, fontsize=25)

    ax[0, 0].set_ylabel("$Z_{pred}$", fontsize=25)

    for i, name in enumerate(preds.keys()):
        x = z_test
        y = (z_test - preds[name]) / (1 + z_test)

        bins = np.linspace(data_lower_lim, data_upper_lim * 1.05, 20)
        x_binned = np.digitize(x, bins)
        y_avg = [y[x_binned == i].mean() for i in range(1, len(bins))]
        y_std = [y[x_binned == i].std() for i in range(1, len(bins))]

        sns.scatterplot(ax=ax[1], x=x, y=y, s=2, alpha=0.3, color="black")
        sns.lineplot(ax=ax[1], x=bins[:-1], y=y_std, color="r", label="std")

        # horizontal line on y = 0
        ax[1].axhline(0, color="grey", linewidth=1.5, alpha=0.5, linestyle="--")

        # sns.scatterplot(ax=ax[1,i], x=bins[:-1], y=y_avg, s=15, color='.15')
        ax[1].set_xlim(data_lower_lim, data_upper_lim)
        ax[1].set_ylim(-data_upper_lim / 2, data_upper_lim / 2)
        ax[1].set_xlabel("$Z_{true}$", fontsize=25)
        ax[1].legend(fontsize=15, loc="upper right")

    ax[1, 0].set_ylabel("$(Z_{true}-Z_{pred})/(1+Z_{true})$", fontsize=25)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_loc, dpi=300)


def plot_scatter_astroclip_only(
    preds: dict,
    z_test: np.ndarray,
    data_lower_lim: float = -1,
    data_upper_lim: float = 0.6,
    save_loc: str = "aclip_scatter.png",
) -> None:
    """Functionality to plot redshift scatter plots for different models."""
    fig, ax = plt.subplots(figsize=(5, 5))

    i=0
    name="AstroCLIP"

    sns.scatterplot(ax=ax, x=z_test, y=preds[name], s=5, color=".15")

    sns.histplot(
        ax=ax, x=z_test, y=preds[name], bins=50, pthresh=0.1, cmap="mako"
    )
    sns.kdeplot(
        ax=ax, x=z_test, y=preds[name], levels=5, color="k", linewidths=1
    )

    ax.plot(
        data_lower_lim,
        data_upper_lim * 1.1,
        "--",
        linewidth=1.5,
        alpha=0.5,
        color="grey",
    )
    ax.set_xlim(data_lower_lim, data_upper_lim)
    ax.set_ylim(data_lower_lim, data_upper_lim)
    ax.text(
        0.9,
        0.1,
        "$R^2$ score: %0.2f" % r2_score(z_test, preds[name]),
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=22,
        transform=ax.transAxes,
    )
    ax.set_title(name, fontsize=25)

    ax.set_ylabel("$KCORR01\_SDSS\_R_{pred}$", fontsize=25)