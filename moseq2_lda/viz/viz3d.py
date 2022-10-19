import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from moseq2_lda.stats import compute_3D_kde, compute_kde
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.proj3d import proj_transform

from moseq2_lda.viz.viz import Aesthetics


class Annotation3D(Annotation):
    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)


def _annotate3D(ax, text, xyz, *args, **kwargs):
    '''Add anotation `text` to an `Axes3d` instance.'''
    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)


setattr(Axes3D, 'annotate3D', _annotate3D)


class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)


def plot_lda_kde_projections_3D(ax, lda_transformed, group_vals, aes: Aesthetics = None, levels: int = 5, alpha: float = 0.5):
    if aes is None:
        aes = Aesthetics(group_vals)

    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    minz, maxz = ax.get_zlim()

    extents = ((minx, maxx), (miny, maxy), (minz, maxz))

    for g in aes.groups:
        mask = np.array(group_vals) == g
        colors = sns.light_palette(aes.palette[aes.groups.index(g)], levels+1)
        for i, c in enumerate(colors):
            colors[i] = (*c, alpha)  # add alpha level to each color level
        colors[0] = (1.0, 1.0, 1.0, 0.0)  # final level should be fully transparent white

        xx, yy, f = compute_kde(lda_transformed, mask, 0, 1, extents)
        ax.contourf(xx, yy, f, zdir='z', offset=minz, colors=colors, levels=levels)

        xx, yy, f = compute_kde(lda_transformed, mask, 0, 2, extents)
        ax.contourf(xx, f, yy, zdir='y', offset=maxy, colors=colors, levels=levels)

        xx, yy, f = compute_kde(lda_transformed, mask, 1, 2, extents)
        ax.contourf(f, xx, yy, zdir='x', offset=minx, colors=colors, levels=levels)

    ax.set_xlim((minx, maxx))
    ax.set_ylim((miny, maxy))
    ax.set_zlim((minz, maxz))


def plot_lda_kde_projections_3D_iso(ax, lda_result, group_vals, aes: Aesthetics, levels: int = 5, alpha: float = 0.5):
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    minz, maxz = ax.get_zlim()

    extents = ((minx, maxx), (miny, maxy), (minz, maxz))

    for g in aes.groups:
        mask = np.array(group_vals) == g
        colors = sns.light_palette(aes.palette[aes.groups.index(g)], levels+1)
        for i, c in enumerate(colors):
            colors[i] = (*c, alpha)  # add alpha level to each color level
        colors[0] = (1.0, 1.0, 1.0, 0.0)  # final level should be fully transparent white

        xx, yy, zz, f = compute_3D_kde(lda_result, mask, extents)
        # ax.plot_surface(xx, yy, zz, )
        ax.contour3D(xx, yy, zz, )

    ax.set_xlim((minx, maxx))
    ax.set_ylim((miny, maxy))
    ax.set_zlim((minz, maxz))


def plot_lda_results_3D(coeff, lda_result, lda_predictions, group_vals, title, figsize, relative_weights, aes: Aesthetics = None):

    if aes is None:
        aes = Aesthetics(group_vals)

    lgd_itms = [mpl.lines.Line2D([0], [0], linestyle="none", c=c, marker=m) for c, m in zip(aes.palette, aes.markers)]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    cs = [aes.palette[aes.groups.index(g)] for g in group_vals]
    ms = [aes.markers[aes.groups.index(g)] for g in group_vals]

    for d, c, m, g in zip(lda_result, cs, ms, group_vals):
        ax.scatter(d[0], d[1], d[2], c=[c], marker=m, label=g)

    fig.legend(lgd_itms, aes.groups)
    ax.set_xlabel('LDA_1')
    ax.set_ylabel('LDA_2')
    ax.set_zlabel('LDA_3')
    ax.set_title(title)

    return fig, ax
