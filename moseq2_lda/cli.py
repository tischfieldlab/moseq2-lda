import click


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
def analyze():
    pass
