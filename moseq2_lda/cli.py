"""CLI interface for the moseq2-lda package."""
import click
import os

import numpy as np

from moseq2_lda.data import load_representations
from moseq2_lda.model import create_lda_pipeline, run_cross_validation, train_lda_model, train_lda_pipeline
from moseq2_lda.util import click_monkey_patch_option_show_defaults
from moseq2_lda.viz import Aesthetics, plot_validation_curve, plot_lda_results, plot_permutation_score


# Show click option defaults
click_monkey_patch_option_show_defaults()


@click.group()
@click.version_option()
def cli():
    """Toolbox for training and using scikit-learn Linear Discriminant Analysis (LDA) models for analysis of moseq data."""
    pass  # pylint: disable=unnecessary-pass


@cli.command(name="create-notebook", help="Create a new jupyter notebook from a template")
@click.argument("dest", required=True, type=click.Path(exists=False))
def create_notebook(dest):
    """WARNING: Not yet implemented!"""
    pass


@cli.command(name="cross-validate", short_help="Run hyperparameter search")
@click.argument("model_file")
@click.argument("index_file")
@click.option("--max-syllable", default=100, help="maximum syllable to consider")
@click.option("--group", type=str, multiple=True, default=None)
@click.option("--representation", type=click.Choice(["usages", "frames", "trans"]), default="usages")
@click.option("--holdout", default=0.3, help="fraction of the data to hold out for testing")
@click.option("--dim", default=2, help="number of dimentions for the LDA model")
@click.option("--dest-dir", type=click.Path(), help="Directory where results will be saved")
@click.option("--name", type=str, default="moseq-lda-analysis", help="basename prefix of output files")
def cross_validate(model_file, index_file, max_syllable, group, representation, holdout, dim, dest_dir, name):
    """Run a cross-validated hyperparameter search for the LDA parameter `shrinkage`."""
    # make sure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    name = f"{name}.{representation}"

    representations = load_representations(index_file, model_file, max_syllable=max_syllable, groups=group)
    representations.describe()

    # Split data into train and test sets.
    # Train will be used for CV and final model training
    # Test will only be used for evaluation of the final model
    train, test = representations.split(holdout)

    # Create a LDA pipeline, passing along any kwargs the user supplied
    estimator = create_lda_pipeline(**{"n_components": dim})

    # Run cross validation using the estimator
    cv_results = run_cross_validation(
        estimator=estimator,
        X=train.data(representation),
        Y=train.groups,
        param_name="shrinkage",
        param_range=[*list(np.linspace(0, 1, 11, dtype=float)), "auto"],
    )

    cv_results.save(os.path.join(dest_dir, f"{name}.cross_validation_results.p"))

    fig, _ = plot_validation_curve(cv_results)
    fig.savefig(os.path.join(dest_dir, f"{name}.validation-curve.png"), dpi=300)
    fig.savefig(os.path.join(dest_dir, f"{name}.validation-curve.pdf"), dpi=300)


@cli.command(name="analyze", short_help="Run default analysis pipeline")
@click.argument("model_file")
@click.argument("index_file")
@click.option("--max-syllable", default=100, help="maximum syllable to consider")
@click.option("--group", type=str, multiple=True, default=None)
@click.option("--representation", type=click.Choice(["usages", "frames", "trans"]), default="usages")
@click.option("--holdout", default=0.3, help="fraction of the data to hold out for testing")
@click.option("--dim", default=2, help="number of dimentions for the LDA model")
@click.option("--dest-dir", type=click.Path(), help="Directory where results will be saved")
@click.option("--name", type=str, default="moseq-lda-analysis", help="basename prefix of output files")
def analyze(model_file, index_file, max_syllable, group, representation, holdout, dim, dest_dir, name):
    r"""Analyze moseq data via LDA.

    \b
    This is a "batteries-included" method which performs the following procedure:
    - split the representations into `test` and `train` subsets
    - creates an LDA estimator
    - run a cross-validated search (k-fold stratified CV) for the hyperparameter `shrinkage` using
        only the `train` subset of the representations
    - select the best hyperparameter value from the search, and then train a classifier using
        the selected hyperparameter value against the full `train` subset of the representations
    - predict on the held-out `test` subsets and print a classification report

    \b
    Outputs:
    - Pickled `LDAPipelineResult` suitable for later loading via `LDAPipelineResult.load()`
    - Plot of the cross-validated hyperparameter search results (png and pdf formats)
    - Plot of LDA results (projections, confusion matrix, table of projections) against
        all data using final model trained from CV
    - Perform and plot permutation test with the final model
    - Classification reports
    """
    # make sure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    name = f"{name}.{representation}"

    representations = load_representations(index_file, model_file, max_syllable=max_syllable, groups=group)
    aes = Aesthetics(group)
    representations.describe()

    results = train_lda_pipeline(representations, representation, holdout=holdout, lda_kwargs={"n_components": dim})
    results.save(os.path.join(dest_dir, f"{name}.lda_pipeline_results.p"))

    fig, _ = plot_validation_curve(results.cv_result)
    fig.savefig(os.path.join(dest_dir, f"{name}.validation-curve.png"), dpi=300)
    fig.savefig(os.path.join(dest_dir, f"{name}.validation-curve.pdf"), dpi=300)

    fig, _, df = plot_lda_results(results.final, representations, aes=aes, title=f"LDA {representation.capitalize()}")
    fig.savefig(os.path.join(dest_dir, f"{name}.lda-results.png"), dpi=300)
    fig.savefig(os.path.join(dest_dir, f"{name}.lda-results.pdf"), dpi=300)
    df.to_csv(os.path.join(dest_dir, f"{name}.lda-results.tsv"), sep="\t", index=False)

    fig, _ = plot_permutation_score(results.final.estimator, results.data.data(representation), results.data.groups)
    fig.savefig(os.path.join(dest_dir, f"{name}.permutation-test.png"), dpi=300)
    fig.savefig(os.path.join(dest_dir, f"{name}.permutation-test.pdf"), dpi=300)

    with open(os.path.join(dest_dir, f"{name}.performance_final-model_training-data.txt"), mode="w") as f:
        f.writelines(results.final.classification_report(results.train))

    with open(os.path.join(dest_dir, f"{name}.performance_final-model_held-out-data.txt"), mode="w") as f:
        f.writelines(results.final.classification_report(results.test))

    with open(os.path.join(dest_dir, f"{name}.performance_final-model_all-data.txt"), mode="w") as f:
        f.writelines(results.final.classification_report(results.data))


@cli.command(name="train", short_help="Train a single LDA model")
@click.argument("model_file")
@click.argument("index_file")
@click.option("--max-syllable", default=100, help="maximum syllable to consider")
@click.option("--group", type=str, multiple=True, default=None)
@click.option("--representation", type=click.Choice(["usages", "frames", "trans"]), default="usages")
@click.option("--holdout", default=0.3, help="fraction of the data to hold out for testing")
@click.option("--dim", default=2, help="number of dimentions for the LDA model")
@click.option("--shrinkage", default="auto", help='float in range of (0, 1] or the string "auto"')
@click.option("--dest-dir", type=click.Path(), help="Directory where results will be saved")
@click.option("--name", type=str, default="moseq-lda-analysis", help="basename prefix of output files")
def train(model_file, index_file, max_syllable, group, representation, holdout, dim, shrinkage, dest_dir, name):
    r"""Train a single LDA model.

    \b
    This command will train a single LDA model
    - split the representations into `test` and `train` subsets
    - creates an LDA estimator, trained on the `train` subset
    - predict on the held-out `test` subsets and print a classification report

    \b
    Outputs:
    - Pickled `LDAResult` suitable for later loading via `LDAResult.load()`
    - Plot of LDA results (projections, confusion matrix, table of projections) against
        all data using the model
    - Perform and plot permutation test with the  \model
    - Classification reports for `test`, `train`, `all` data subsets
    """
    if not shrinkage == "auto":
        shrinkage = float(shrinkage)

    # make sure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    shrink_str = f"{shrinkage:0.3f}" if isinstance(shrinkage, float) else str(shrinkage)
    name = f"{name}.{representation}.{dim}D_{shrink_str}-shrink"

    representations = load_representations(index_file, model_file, max_syllable=max_syllable, groups=group)
    aes = Aesthetics(group)
    representations.describe()

    train, test = representations.split(holdout)

    # Create a LDA pipeline, passing along with any parameters the user supplied
    estimator = create_lda_pipeline(**{"n_components": dim, "shrinkage": shrinkage})

    # train the model on the train subset
    final = train_lda_model(estimator, train, representation)
    final.save(os.path.join(dest_dir, f"{name}.lda_results.p"))

    with open(os.path.join(dest_dir, f"{name}.performance_final-model_training-data.txt"), mode="w") as f:
        f.writelines(final.classification_report(train))

    with open(os.path.join(dest_dir, f"{name}.performance_final-model_held-out-data.txt"), mode="w") as f:
        f.writelines(final.classification_report(test))

    with open(os.path.join(dest_dir, f"{name}.performance_final-model_all-data.txt"), mode="w") as f:
        f.writelines(final.classification_report(representations))

    # plot the results
    fig, _, df = plot_lda_results(final, representations, aes=aes, title=f"LDA {representation.capitalize()}")
    fig.savefig(os.path.join(dest_dir, f"{name}.lda-results.png"), dpi=300)
    fig.savefig(os.path.join(dest_dir, f"{name}.lda-results.pdf"), dpi=300)
    df.to_csv(os.path.join(dest_dir, f"{name}.lda-results.tsv"), sep="\t", index=False)

    # run and plot permutation test
    fig, _ = plot_permutation_score(final.estimator, final.data.data(representation), final.data.groups)
    fig.savefig(os.path.join(dest_dir, f"{name}.permutation-test.png"), dpi=300)
    fig.savefig(os.path.join(dest_dir, f"{name}.permutation-test.pdf"), dpi=300)


if __name__ == "__main__":
    cli()
