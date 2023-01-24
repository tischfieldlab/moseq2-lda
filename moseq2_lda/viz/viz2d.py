from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from moseq2_lda.util import dict_merge
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import matplotlib as mpl

from moseq2_lda.viz.viz import Aesthetics


def plot_lda_arrows(ax, lda_result, group_vals, arrow_specs):

    for ag in arrow_specs:
        args = {
            "padding": 0.7,
            "kwargs": {
                "alpha": 0.5,
            },
        }
        dict_merge(args, ag)

        g1_c = np.mean(lda_result.T[:, np.array(group_vals) == args["from"]], axis=1)
        g2_c = np.mean(lda_result.T[:, np.array(group_vals) == args["to"]], axis=1)

        d = (g2_c - g1_c) * args["padding"]
        c = (g1_c + g2_c) / 2
        r = c - (d / 2)

        if r.shape[0] == 3:
            if "mutation_scale" not in args["kwargs"]:
                args["kwargs"]["mutation_scale"] = 20
            ax.arrow3D(x=r[0], y=r[1], z=r[2], dx=d[0], dy=d[1], dz=d[2], **args["kwargs"])
        else:
            ax.arrow(x=r[0], y=r[1], dx=d[0], dy=d[1], length_includes_head=True, **args["kwargs"])


# style:Literal['kde', 'centroid']
def plot_lda_projection(lda_transformed, group_vals, groups, palette, markers, title="LDA", ax=None, style=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if style == "kde":
        try:
            sns.kdeplot(
                x=lda_transformed[:, 0],
                y=lda_transformed[:, 1],
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
        sns.scatterplot(
            x=lda_transformed[:, 0],
            y=lda_transformed[:, 1],
            hue=group_vals,
            hue_order=groups,
            style=group_vals,
            style_order=groups,
            markers=markers,
            palette=palette,
            legend=False,
            ax=ax,
        )

    elif style == "centroid":
        desat_palette = sns.color_palette(palette, desat=0.3)
        sns.scatterplot(
            x=lda_transformed[:, 0],
            y=lda_transformed[:, 1],
            hue=group_vals,
            hue_order=groups,
            style=group_vals,
            style_order=groups,
            markers=markers,
            palette=desat_palette,
            legend=False,
            alpha=0.5,
            ax=ax,
        )
        gmeans = []
        for g in groups:
            gmeans.append(lda_transformed[group_vals == g].mean(axis=0))
        gmeans = np.array(gmeans)
        # print(lda_transformed, group_vals, groups, gmeans, gmeans.shape, group_vals == g)
        ax = sns.scatterplot(
            x=gmeans[:, 0],
            y=gmeans[:, 1],
            hue=groups,
            hue_order=groups,
            style=groups,
            style_order=groups,
            markers=markers,
            palette=palette,
            legend="full",
            alpha=1.0,
            s=420,
            ax=ax,
        )

    else:
        sns.scatterplot(
            x=lda_transformed[:, 0],
            y=lda_transformed[:, 1],
            hue=group_vals,
            hue_order=groups,
            style=group_vals,
            style_order=groups,
            markers=markers,
            palette=palette,
            legend=False,
            ax=ax,
        )

    ax.set_xlabel("LDA_1")
    ax.set_ylabel("LDA_2")
    ax.set_title(title)

    return ax


def plot_lda_confusion_matrix(lda_predictions, group_vals, groups, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    confusion = confusion_matrix(y_true=group_vals, y_pred=lda_predictions, normalize="true", labels=groups)
    sns.heatmap(
        confusion, ax=ax, xticklabels=groups, yticklabels=groups, cbar_kws={"label": "Row Normalized Confusion"}, vmin=0, vmax=1, annot=True
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("LDA Classification Confusion Matrix")

    return ax


def plot_lda_weights(lda_coeff, groups, relative_weights=None, ax=None):
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


def plot_lda_scalings(lda, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    # normalize
    lda_coeff = lda.scalings_
    lda_coeff = lda_coeff / np.max(np.abs(lda_coeff), axis=0)

    palette = plt.get_cmap("jet", lda_coeff.shape[1])  # flare

    # Plot a variable factor map for the first two dimensions.
    # (fig, ax) = plt.subplots(figsize=(8, 8))
    for i in range(0, lda_coeff.shape[1]):
        ax.arrow(
            0,
            0,  # Start the arrow at the origin
            lda_coeff[0, i],  # 0 for PC1
            lda_coeff[1, i],  # 1 for PC2
            head_width=0.05,
            head_length=0.05,
            color=palette(i),
        )

        plt.text(lda_coeff[0, i] + 0.05, lda_coeff[1, i] + 0.05, i)

    norm = mpl.colors.Normalize(vmin=0, vmax=lda_coeff.shape[1])
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    sm.set_array([])
    cax = inset_axes(
        ax, width="3%", height="100%", loc="lower left", bbox_to_anchor=(1.05, 0, 1, 1), bbox_transform=ax.transAxes, borderpad=0
    )
    plt.colorbar(sm, cax=cax, label="variable")

    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))
    ax.set_xlabel(f"LDA_1 ({lda.explained_variance_ratio_[0]:0.1%})")
    ax.set_ylabel(f"LDA_2 ({lda.explained_variance_ratio_[1]:0.1%})")
    # an = np.linspace(0, 2 * np.pi, 100)
    # plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
    # plt.axis('equal')
    ax.set_title("Variable factor map")

    return ax


def plot_lda_results_2D(
    lda, lda_transformed, lda_predictions, group_vals, aes: Aesthetics = None, title="LDA", figsize=(25, 25), relative_weights=None
):

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

    plot_lda_scalings(lda, ax=axs[1, 1])

    return fig, axs
