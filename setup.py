#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for the pyexodus module.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2016
:license:
    MIT License
"""
import inspect
import os
import sys

from setuptools import setup, find_packages


# Import the version string.
path = os.path.join(
    os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe()))),
    "pyexodus",
)
sys.path.insert(0, path)
from version import get_git_version  # NOQA


def get_package_data():
    """
    Returns a list of all files needed for the installation relative to the
    'pyexodus' subfolder.
    """
    filenames = []
    # The lasif root dir.
    root_dir = os.path.join(
        os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        ),
        "pyexodus",
    )
    # Recursively include all files in these folders:
    folders = [os.path.join(root_dir, "tests", "data")]
    for folder in folders:
        for directory, _, files in os.walk(folder):
            for filename in files:
                # Exclude hidden files.
                if filename.startswith("."):
                    continue
                filenames.append(
                    os.path.relpath(
                        os.path.join(directory, filename), root_dir
                    )
                )
    return filenames


setup_config = dict(
    name="pyexodus",
    version=get_git_version(),
    description="Module for creating Exodus files",
    long_description=(
        "Create Exodus files with Python "
        "(https://salvushub.github.io/pyexodus/)"
    ),
    author="Lion Krischer",
    author_email="lionkrischer@gmail.com",
    url="https://github.com/SalvusHub/pyexodus",
    packages=find_packages(),
    license="MIT",
    platforms="OS Independent",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=["numpy", "h5netcdf >= 0.5.0"],
    package_data={"pyexodus": get_package_data()},
)


if __name__ == "__main__":
    setup(**setup_config)
