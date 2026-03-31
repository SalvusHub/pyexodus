#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (lionkrischer@gmail.com), 2016
:license:
    MIT License
"""
from .core import exodus  # NOQA

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"
