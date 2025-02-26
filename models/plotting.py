import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score

def plot_scatter(
    preds: np.ndarray,
    ground_truth: np.ndarray,
    data_lower_lim: float = -0.5,
    data_upper_lim: float = 0.5,
    varname = "KCORR01\_SDSS\_R"
) -> None:
    """Functionality to plot redshift scatter plots for different models."""
    fig, ax = plt.subplots(figsize=(5, 5))

    i=0
    name="AstroCLIP"

    sns.scatterplot(ax=ax, x=ground_truth, y=preds, s=5, color=".15")

    sns.histplot(
        ax=ax, x=ground_truth, y=preds, bins=50, pthresh=0.1, cmap="mako"
    )
    sns.kdeplot(
        ax=ax, x=ground_truth, y=preds, levels=5, color="k", linewidths=1
    )

    ax.plot(
        data_lower_lim,
        data_upper_lim * 1.1,
        "--",
        linewidth=1.5,
        alpha=0.5,
        color="grey",
    )

    ax.plot([data_lower_lim, data_upper_lim], [data_lower_lim, data_upper_lim], '--', color='red', linewidth=1)
    ax.set_xlim(data_lower_lim, data_upper_lim)
    ax.set_ylim(data_lower_lim, data_upper_lim)

    ax.set_title(name, fontsize=25)

    ax.set_ylabel(varname, fontsize=25)