import click
import os

from moseq2_lda.data import load_representations
from moseq2_lda.model import train_lda_pipeline
from moseq2_lda.viz import Aesthetics, plot_validation_curve, plot_lda_results, plot_permutation_score


@click.group()
@click.version_option()
def cli():
    ''' Toolbox for training and using scikit-learn Linear Discriminant Analysis models
        for analysis of moseq models
    '''
    pass  # pylint: disable=unnecessary-pass


@cli.command(name='create-notebook', help='Create a new jupyter notebook from a template')
@click.argument('dest', required=True, type=click.Path(exists=False))
def create_notebook(dest):
    pass


@cli.command(name='analyze', help='Run default analysis pipeline')
@click.argument('model_file')
@click.argument('index_file')
@click.option('--max-syllable', default=100, help='maximum syllable to consider')
@click.option('--group', type=str, multiple=True, default=None)
@click.option('--representation', type=click.Choice(['usages', 'frames', 'trans']), default='usages')
@click.option('--dest-dir', type=click.Path(), help='Directory where results will be saved')
@click.option('--name', type=str, default='moseq-lda-analysis', help='basename prefix of output files')
def analyze(model_file, index_file, max_syllable, group, representation, dest_dir, name):
    ''' Analyze moseq data via LDA '''

    # make sure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    name = f'{name}.{representation}'

    representations = load_representations(index_file, model_file, max_syllable=max_syllable, groups=group)
    aes = Aesthetics(group)
    representations.describe()

    results = train_lda_pipeline(representations, representation, lda_kwargs={'n_components': 2})
    results.save(os.path.join(dest_dir, f'{name}.lda_pipeline_results.p'))

    fig, _ = plot_validation_curve(results.cv_result)
    fig.savefig(os.path.join(dest_dir, f'{name}.validation-curve.png'), dpi=300)
    fig.savefig(os.path.join(dest_dir, f'{name}.validation-curve.pdf'), dpi=300)

    fig, _, df = plot_lda_results(results.final, representations, aes=aes, title=f'LDA {representation.capitalize()}')
    fig.savefig(os.path.join(dest_dir, f'{name}.lda-results.png'), dpi=300)
    fig.savefig(os.path.join(dest_dir, f'{name}.lda-results.pdf'), dpi=300)
    df.to_csv(os.path.join(dest_dir, f'{name}.lda-results.tsv'), sep='\t', index=False)

    fig, _ = plot_permutation_score(results.final.estimator, results.data.data(representation), results.data.groups)
    fig.savefig(os.path.join(dest_dir, f'{name}.permutation-test.png'), dpi=300)
    fig.savefig(os.path.join(dest_dir, f'{name}.permutation-test.pdf'), dpi=300)

    with open(os.path.join(dest_dir, f'{name}.performance_final-model_held-out-data.txt'), mode='w') as f:
        f.writelines(results.final.classification_report(results.test))

    with open(os.path.join(dest_dir, f'{name}.performance_final-model_all-data.txt'), mode='w') as f:
        f.writelines(results.final.classification_report(results.data))


if __name__ == '__main__':
    cli()
