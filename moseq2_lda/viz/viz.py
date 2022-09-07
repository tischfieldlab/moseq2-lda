from typing import List
from dataclasses import asdict

import pandas as pd
from matplotlib import pyplot as plt
from moseq2_lda.model import CrossValidationResult
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import permutation_test_score
import seaborn as sns

class PlotAesthetics:

    default_palette = 'deep'
    marker_pool = ['o', 'v', '^', '<', '>', 's', 'p', 'P', 'D', 'X', '*', 'h', 'H', 'd']

    def __init__(self, groups: List[str]):
        self.groups = list(set(groups))
        n_groups = len(self.groups)
        self.palette = sns.color_palette(self.default_palette, n_colors=n_groups)
        self.markers = [self.marker_pool[i % len(self.marker_pool)] for i in range(n_groups)]




def plot_validation_curve(cv_result: CrossValidationResult, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.set_title('Validation Curve')
    ax.set_xlabel(cv_result.param_name)
    ax.set_ylabel(f'Mean {cv_result.scoring}')
    ax.set_ylim(0.0, 1.1)
    #ax.set_xlim(cv_result.param_min, cv_result.param_max)
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

    return ax



def plot_lda_results(lda, data, meta_vals, group_vals, groups, palette, markers, title="LDA", figsize=(25, 20), relative_weights=None):

    lda_result = lda.transform(data)
    lda_predictions = lda.predict(data)
    
    if isinstance(lda, Pipeline):
        lda_ = lda[-1]
    else:
        lda_ = lda


    print('LDA Score: {}'.format(lda.score(data, group_vals)))
    print('LDA Explained Variance: {}'.format(lda_.explained_variance_ratio_))
    print(classification_report(y_true=group_vals, y_pred=lda_predictions))

    out_data = []
    for i in range(len(group_vals)):
        out_data.append({
            **asdict(meta_vals[i]),
            'group': group_vals[i],
            'predicted_group': lda_predictions[i],
            **{f'LDA_{d+1}': lda_result[i, d] for d in range(lda_.n_components)}
        })
    out_data = pd.DataFrame(out_data)


    if lda_.n_components == 2:
        from .viz2d import plot_lda_results_2D
        fig, axs = plot_lda_results_2D(lda_, lda_result, lda_predictions, group_vals, groups, palette, markers, title, figsize, relative_weights)
    elif lda_.n_components == 3:
        from .viz3d import plot_lda_kde_projections_3D, plot_lda_results_3D
        fig, axs = plot_lda_results_3D(lda_, lda_result, lda_predictions, group_vals, groups, palette, markers, title, figsize, relative_weights)
        plot_lda_kde_projections_3D(axs, lda_result, groups, group_vals, palette)
    else:
        raise ValueError(f'unsupported `n_components` of {lda_.n_components}; only 2-3 components allowed!')

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

    return ax
