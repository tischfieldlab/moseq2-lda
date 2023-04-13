
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_lda_confusion_matrix(lda_predictions, group_vals, groups, ax=None):
    """Plot a confusion matrix for LDA classifications."""
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
