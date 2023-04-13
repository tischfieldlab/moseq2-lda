"""Module contains 2D-specific plotting functions."""
from typing import Optional
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from moseq2_lda.util import dict_merge
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import matplotlib as mpl
from moseq2_lda.viz.common import plot_lda_confusion_matrix

from moseq2_lda.viz.viz import Aesthetics





# style:Literal['kde', 'centroid']
def plot_lda_projection(lda_transformed, group_vals, groups, palette, markers, title="LDA", ax=None, style=None):
    """Plot a projection of the LDA data."""
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if style == "kde":
        try:
            sns.kdeplot(
                x=lda_transformed[:, 0],
                hue=group_vals,
                hue_order=groups,
                palette=palette,
                fill=True,
                alpha=0.2,
                ax=ax,
                legend=False,
            )
        except Exception as e:
            print("Warn: Failed to draw KDE")
            print(e)
            pass

        sns.rugplot(
            x=lda_transformed[:, 0],
            hue=group_vals,
            hue_order=groups,
            palette=palette,
            legend=False,
            ax=ax,
        )

        ax.set_xlabel("LDA_1")
        ax.set_ylabel("Density")

    elif style == "centroid":
        desat_palette = sns.color_palette(palette, desat=0.3)
        sns.boxplot(
            x=group_vals,
            y=lda_transformed[:, 0],
            hue=group_vals,
            hue_order=groups,
            palette=palette,
            ax=ax,
        )
        sns.swarmplot(
            x=group_vals,
            y=lda_transformed[:, 0],
            hue=group_vals,
            hue_order=groups,
            palette=desat_palette,
            ax=ax,
        )

        ax.set_ylabel("LDA_1")
        ax.set_xlabel("Group")
    else:
        sns.swarmplot(
            y=lda_transformed[:, 0],
            hue=group_vals,
            hue_order=groups,
            palette=palette,
            legend=False,
            ax=ax,
        )
        ax.set_ylabel("LDA_1")
        ax.set_xlabel("Group")


    ax.set_title(title)

    return ax


def plot_lda_weights(lda_coeff, groups, relative_weights=None, ax=None):
    """Plot weights for an LDA model."""
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    coef_heat_kwargs = {"ax": ax, "xticklabels": groups, "cbar_kws": {"label": "Feature Weights"}}
    if relative_weights is not None:
        weights = (lda_coeff - lda_coeff[groups.index(relative_weights)]).T
    else:
        weights = lda_coeff.T

    if np.min(weights) < 0:
        # positive and negative values
        coef_heat_kwargs.update({"cmap": "vlag", "center": 0})
    else:
        # only positive values
        coef_heat_kwargs.update({"cmap": "rocket", "center": None})

    sns.heatmap(weights, **coef_heat_kwargs)
    ax.set_xlabel("Class")
    ax.set_ylabel("Feature")
    ax.set_title("LDA Feature Weights")

    return ax


def plot_lda_results_1D(
    lda,
    lda_transformed,
    lda_predictions,
    group_vals,
    aes: Optional[Aesthetics] = None,
    title="LDA",
    figsize=(25, 25),
    relative_weights=None,
):
    """Plot results for a 2D LDA model."""
    if aes is None:
        aes = Aesthetics(group_vals)

    fig, axs = plt.subplots(2, 2, figsize=figsize)

    plot_lda_projection(
        lda_transformed, group_vals, aes.groups, palette=aes.palette, markers=aes.markers, title=title, ax=axs[0, 0], style="kde"
    )

    plot_lda_projection(
        lda_transformed, group_vals, aes.groups, palette=aes.palette, markers=aes.markers, title=title, ax=axs[0, 1], style="centroid"
    )

    fig.subplots_adjust(right=0.75)
    sns.move_legend(axs[0, 1], "upper left", bbox_to_anchor=(1.05, 0.5))
    sns.despine(fig=fig)

    plot_lda_confusion_matrix(lda_predictions, group_vals, aes.groups, ax=axs[1, 0])

    #plot_lda_scalings(lda, ax=axs[1, 1])

    return fig, axs
