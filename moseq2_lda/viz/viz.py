from dataclasses import asdict
from typing import ClassVar, List

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from moseq2_lda.data import MoseqRepresentations
from moseq2_lda.model import CrossValidationResult, LdaResult
from sklearn.model_selection import permutation_test_score


class Aesthetics:

    default_palette: ClassVar[str] = 'deep'
    marker_pool: ClassVar[List[str]] = ['o', 'v', '^', '<', '>', 's', 'p', 'P', 'D', 'X', '*', 'h', 'H', 'd']

    def __init__(self, groups: List[str], palette=None, markers=None):
        self.groups = list(dict.fromkeys(groups))
        n_groups = len(self.groups)

        if palette is None:
            palette = self.default_palette
        self.palette = sns.color_palette(palette, n_colors=n_groups)

        if markers is None:
            self.markers = [self.marker_pool[i % len(self.marker_pool)] for i in range(n_groups)]
        else:
            self.markers = markers


def plot_validation_curve(cv_result: CrossValidationResult, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.set_title('Validation Curve')
    ax.set_xlabel(cv_result.param_name)
    ax.set_ylabel(f'Mean {cv_result.scoring}')
    ax.set_ylim(0.0, 1.1)
    # ax.set_xlim(cv_result.param_min, cv_result.param_max)
    ax.plot(cv_result.param_range, cv_result.train_scores_mean, label='Training score ± stdev', color='darkorange')
    ax.fill_between(
        range(len(cv_result.param_range)),
        cv_result.train_scores_mean - cv_result.train_scores_std,
        cv_result.train_scores_mean + cv_result.train_scores_std,
        alpha=0.2,
        color='darkorange',
        lw=2,
    )

    ax.plot(cv_result.param_range, cv_result.test_scores_mean, label='Cross-validation score ± stdev', color='navy')
    ax.fill_between(
        range(len(cv_result.param_range)),
        cv_result.test_scores_mean - cv_result.test_scores_std,
        cv_result.test_scores_mean + cv_result.test_scores_std,
        alpha=0.2,
        color='navy',
        lw=2,
    )
    best = cv_result.best
    best_val_str = best["param"] if isinstance(best["param"], str) else f'{best["param"]:0.2f}'
    ax.axvline(x=best['idx'], label=f'Best `{cv_result.param_name}` value ({best_val_str})', linestyle='--', color='red')
    ax.axhline(y=cv_result.chance, label=f'Chance ({cv_result.chance:0.3f})', linestyle='--', color='gray')

    tick_labels = [x if isinstance(x, str) else f'{x:0.2f}' for x in cv_result.param_range]
    ax.set_xticks(range(len(cv_result.param_range)))
    ax.set_xticklabels(tick_labels)

    ax.legend(loc='best')

    return fig, ax


def plot_lda_results(lda: LdaResult, data: MoseqRepresentations, aes: Aesthetics = None, title="LDA",
                     figsize=(25, 20), relative_weights=None):

    lda_result = lda.transform(data)
    lda_predictions = lda.predict(data)

    print('LDA Score: {}'.format(lda.score(data)))
    print('LDA Explained Variance: {}'.format(lda.lda.explained_variance_ratio_))
    print(lda.classification_report(data))

    out_data = []
    for i, m in enumerate(data.meta):
        out_data.append({
            **asdict(m),
            'predicted_group': lda_predictions[i],
            **{f'LDA_{d+1}': lda_result[i, d] for d in range(lda.lda.n_components)}
        })
    out_data = pd.DataFrame(out_data)

    if lda.lda.n_components == 2:
        from .viz2d import plot_lda_results_2D
        fig, axs = plot_lda_results_2D(lda.lda, lda_result, lda_predictions, data.groups, aes=aes, title=title, figsize=figsize,
                                       relative_weights=relative_weights)
    elif lda.lda.n_components == 3:
        from .viz3d import plot_lda_kde_projections_3D, plot_lda_results_3D
        fig, axs = plot_lda_results_3D(lda.lda, lda_result, lda_predictions, data.groups, aes=aes, title=title, figsize=figsize,
                                       relative_weights=relative_weights)
        plot_lda_kde_projections_3D(axs, lda_result, data.groups, aes=aes)
    else:
        raise ValueError(f'unsupported `n_components` of {lda.lda.n_components}; only 2-3 components allowed!')

    return fig, axs, out_data


def plot_permutation_score(lda, X, y, cv=None, n_permutations=1000, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    score, perm_scores, pvalue = permutation_test_score(
        lda, X, y, scoring="accuracy", cv=cv, n_permutations=n_permutations
    )

    ax.hist(perm_scores, bins=20, density=True, label=f'Randomized Data Scores (n={n_permutations})')
    ax.axvline(score, ls="--", color="r", label=f'Real Data Score ({score:.2f})')
    ax.axvline(pvalue, ls=":", color="k", label=f'p-value ({pvalue:.3E})')
    ax.set_xlabel("Accuracy score")
    ax.set_ylabel("Probability")

    ax.legend(loc='best')

    return fig, ax


def save_figure(fig, dest, format):
    pass
