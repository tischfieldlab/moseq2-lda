from distutils.core import setup
from distutils.util import convert_path
import setuptools
import os
from pathlib import Path



setup(
    name="moseq2-lda",
    license="MIT License",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "tqdm",
        "h5py",
        "matplotlib",
        "seaborn",
        "ruamel.yaml",
        "moseq2-viz",
    ],
    description="Updated version of moseq2 with graphs and table",
    packages=setuptools.find_packages(),
    include_package_data=True,
    #entry_points={"console_scripts": scripts, },
    zip_safe=False,
)
