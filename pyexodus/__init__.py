#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (lion.krischer@mondaic.com), 2016-2026
    Mondaic Ltd. (info@mondaic.com), 2020-2026
:license:
    MIT License
"""
from importlib.metadata import version as _get_version

from .core import exodus  # NOQA

__version__ = _get_version("pyexodus")