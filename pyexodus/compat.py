#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (lionkrischer@gmail.com), 2022
:license:
    MIT License
"""
import numpy as np


def get_dim_size(f, dimension_name):
    """
    Compatibility for h5netcdf<0.14.0 and newer versions.

    :param f: The parameter.
    :param dimension_name: The name of the dimension.
    """
    dim = f.dimensions[dimension_name]
    # Used to be an integer.
    if isinstance(dim, (int, np.integer)):
        return dim
    elif dim is None:
        return dim
    # An object in newer version.
    elif dim.isunlimited():
        return None
    else:
        return dim.size
