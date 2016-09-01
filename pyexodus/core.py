#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (lionkrischer@gmail.com), 2016
:license:
    MIT License
"""
from __future__ import absolute_import

import os
import numpy as np

import h5netcdf


class exodus(object):
    def __init__(self, file, mode="r", array_type="numpy", title=None,
                 numDims=None, numNodes=None, numElems=None, numBlocks=None,
                 numNodeSets=None, numSideSets=None, io_size=0):

        # API is currently quite limited...mainly because nothing else is
        # implemented.
        assert mode == "w", "Currently only writing is supported."
        assert array_type == "numpy", "array_type must be 'numpy'."
        assert numBlocks == 1, "numBlocks must be 1 for now."
        assert numNodeSets == 0, "numNodeSets must be 1 for now."
        assert io_size == 0, "io_size must be 0 for now."
        assert numDims in [2, 3], "Only 2 or 3 dimensions are supported."

        assert not os.path.exists(file), "File '%s' already exists." % file

        self._f = h5netcdf.File(file)

        self._write_attrs(title=title)

        # Set the dimensions - very straighforward.
        self._f.dimensions = {
            # XXX: These should come from some header variable.
            "four": 4,  # No clue what the purpose of this is...
            "len_line": 81,
            "len_name": 33,
            "len_string": 33,
            # These are dynamic.
            "num_dim": numDims,
            "num_el_blk": numBlocks,
            "num_elem": numElems,
            "num_nodes": numNodes,
            "num_side_sets": numSideSets,
            # XXX: Currently must be set to one as h5netcdf does currently
            # not support unlimited dimensions altough this should be easy
            # to add.
            "time_step": 1}

        self._create_variables()

    def put_info_records(self, strings):
        # XXX: Currently a no-op as our examples don't really use it.
        assert not strings, "Not yet implemented."

    def _write_attrs(self, title):
        """
        Write all the attributes.
        """
        # XXX: Should probably all be defined in some header file.
        self._f.attrs['api_version'] = \
            np.array([6.30000019], dtype=np.float32),
        self._f.attrs['file_size'] = np.array([1], dtype=np.int32),
        self._f.attrs['floating_point_word_size'] = \
            np.array([8], dtype=np.int32),
        self._f.attrs['int64_status'] = np.array([0], dtype=np.int32),
        self._f.attrs['maximum_name_length'] = np.array([32], dtype=np.int32)
        self._f.attrs['title'] = title
        self._f.attrs['version'] = np.array([6.30000019], dtype=np.float32)

    def _create_variables(self):
        # Coordinate names.
        self._f.create_variable('/coor_names', ('num_dim', 'len_name'),
                                dtype='|S1')

        # Coordinates.
        _coords = "xyz"
        for _i in range(self._f.dimensions["num_dim"]):
            self._f.create_variable(
                '/coord' + _coords[_i], ('num_nodes',), dtype=np.float64)

        # ??
        self._f.create_variable('/eb_names', ('num_el_blk', 'len_name'),
                                dtype='|S1')
        self._f.create_variable('/eb_prop1', ('num_el_blk',),
                                dtype=np.int32, data=[-1])
        self._f.variables["eb_prop1"].attrs['name'] = 'ID'
        self._f.create_variable('/eb_status', ('num_el_blk',), dtype=np.int32)

        # Side sets.
        self._f.create_variable('/ss_names', ('num_side_sets', 'len_name'),
                                dtype='|S1')
        self._f.create_variable('/ss_prop1', ('num_side_sets',),
                                dtype=np.int32, data=[-1])
        self._f.variables["ss_prop1"].attrs['name'] = 'ID'
        self._f.create_variable('/ss_status', ('num_side_sets',),
                                dtype=np.int32)

        # Time steps.
        self._f.create_variable('/time_whole', ('time_step',),
                                dtype=np.float64)

    def __del__(self):
        try:
            self._f.close()
        except:
            pass

    def close(self):
        try:
            self._f.close()
        except:
            pass

    def __enter__(self):
        """
        Enable usage as a context manager.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Enable usage as a context manager.
        """
        self.__del__()
