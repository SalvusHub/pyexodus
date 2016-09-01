#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (lionkrischer@gmail.com), 2016
:license:
    MIT License
"""
from __future__ import absolute_import

import re
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

        # Set the dimensions - very straightforward.
        # XXX: These should come from some header variable.
        self._f.dimensions["len_string"] = 33
        self._f.dimensions["len_line"] = 81
        # No clue what this is for...
        self._f.dimensions["four"] = 4
        self._f.dimensions["len_name"] = 33

        # XXX: Currently must be set to one as h5netcdf does currently
        # not support unlimited dimensions altough this should be easy
        # to add.
        self._f.dimensions["time_step"] = 1

        # These are dynamic.
        self._f.dimensions["num_dim"] = numDims
        self._f.dimensions["num_nodes"] = numNodes
        self._f.dimensions["num_elem"] = numElems
        self._f.dimensions["num_el_blk"] = numBlocks
        if numSideSets:
            self._f.dimensions["num_side_sets"] = numSideSets

        self._create_variables()

    def put_info_records(self, strings):
        # XXX: Currently a no-op as our examples don't really use it.
        assert not strings, "Not yet implemented."

    def put_coords(self, xCoords, yCoords, zCoords):
        """
        Put coordinates in the exodus file.
        """
        self._f.variables["coordx"][:] = xCoords
        self._f.variables["coordy"][:] = yCoords
        self._f.variables["coordz"][:] = zCoords

    def put_elem_blk_info(self, id, elemType, numElems, numNodesPerElem,
                          numAttrsPerElem):
        """
        Set the details of an element block.

        :type id: int
        :param id: The id of the element block.
        :type elemType: str
        :param elemType: The name of the element block.
        :type numElems: int
        :param numElems: The number of elements in the block.
        :type numNodesPerElem: int
        :param numNodesPerElem: The number of nodes per element.
        :type numAttrsPerElem: int
        :param numAttrsPerElem: The number of attributes per element.
        """
        assert numElems <= self._f.dimensions["num_elem"], \
            "Canont have more elements in the block then globally set."
        assert numAttrsPerElem == 0, "Must be 0 for now."

        num_el_name = "num_el_in_blk%i" % id
        num_node_per_el_name = "num_nod_per_el%i" % id
        var_name = "connect%i" % id

        self._f.dimensions[num_el_name] = numElems
        self._f.dimensions[num_node_per_el_name] = numNodesPerElem

        self._f.create_variable(
            var_name, (num_el_name, num_node_per_el_name),
            dtype=np.int32)
        self._f.variables[var_name].attrs['elem_type'] = np.string_(elemType)

        self._f.variables['eb_status'][:] += 1

    def put_elem_connectivity(self, id, connectivity):
        """
        Set the element connectivity array for all elements in a block.

        The information for this block must have already been set by calling
        e.put_elem_blk_info() on this block.

        :type id: int
        :param id: The id of the element block.
        :type connectivity: :class:`numpy.ndarray`
        :param connectivity: The connectivity. Must be equal to the number
            of of elements times the number of nodes per element for any
            given block.
        """
        num_el_name = "num_el_in_blk%i" % id
        num_node_per_el_name = "num_nod_per_el%i" % id
        var_name = "connect%i" % id

        assert num_el_name in self._f.dimensions, \
            "Block id %i does not exist" % id
        assert num_node_per_el_name in self._f.dimensions, \
            "Block id %i does not exist" % id

        assert connectivity.shape == (
            self._f.dimensions[num_el_name] *
            self._f.dimensions[num_node_per_el_name],)

        self._f.variables[var_name][:] = \
            connectivity.reshape((self._f.dimensions[num_el_name],
                                  self._f.dimensions[num_node_per_el_name]))

    def put_time(self, step, value):
        """
        Put time step and value into exodus file.

        :type step: int
        :param step: The index of the time step. First is 1.
        :type value: float
        :param value: The actual time at that index.
        """
        assert step > 0, "Step must be larger than 0."
        # XXX: Currently the time axis is not unlimited due to a limitation
        # in h5netcdf - thus no new time steps can be created after the
        # initialization.
        assert step <= self._f.dimensions["time_step"]

        self._f.variables["time_whole"][step - 1] = value

    def set_global_variable_number(self, number):
        """
        Set number of global variables in exodus file.

        :type number: int
        :param number: The number of global variables.
        """
        if not number:  # pragma: no cover
            return

        self._f.dimensions["num_glo_var"] = number

        self._f.create_variable(
            "name_glo_var", ("num_glo_var", "len_name"),
            dtype="|S1")
        self._f.create_variable(
            "vals_glo_var", ("time_step", "num_glo_var"),
            dtype=np.float64)

    def put_global_variable_name(self, name, index):
        """
        Put global variable with name at index into exodus file.

        :type name: str
        :param name: The name of the variable.
        :type index: int
        :param index: The index of the global variable. First is 1!
        """
        self._f.variables["name_glo_var"][index - 1] = ""
        self._f.variables["name_glo_var"][index - 1, :len(name)] = list(name)

    def put_global_variable_value(self, name, step, value):
        """
        Put global variable value and variable name at time step into exodus
        file.

        :type name: str
        :param name: The name of the variable.
        :type step: int
        :param step: The index of the time step. First is 1.
        :type value: float
        :param value: The actual time at that index.
        """
        assert step > 0, "Step must be larger than 0."
        # XXX: Currently the time axis is not unlimited due to a limitation
        # in h5netcdf - thus no new time steps can be created after the
        # initialization.
        assert step <= self._f.dimensions["time_step"]

        idx = self.get_global_variable_names().index(name)
        self._f.variables["vals_glo_var"][step - 1, idx] = value

    def get_global_variable_names(self):
        """
        Get list of global variable names in exodus file.
        """
        return ["".join(_i).strip()
                for _i in self._f.variables["name_glo_var"][:]]

    def set_element_variable_number(self, number):
        """
        Set number of element variables in exodus file.

        :type number: int
        :param number: The number of variables per element.
        """
        self._f.dimensions["num_elem_var"] = number

        self._f.create_variable(
            "name_elem_var", ("num_elem_var", "len_name"),
            dtype="|S1")

    def put_element_variable_name(self, name, index):
        """
        Element variable with name at index into exodus file.

        :type name: str
        :param name: The name of the element variable.
        :type index: int
        :param index: The index of the element variable. Starts with 1!
        """
        self._f.variables["name_elem_var"][index - 1] = ""
        self._f.variables["name_elem_var"][index - 1, :len(name)] = \
            list(name)

    def get_element_variable_names(self):
        """
        Get list of element variable names in exodus file.
        """
        return ["".join(_i).strip()
                for _i in self._f.variables["name_elem_var"][:]]

    def put_element_variable_values(self, blockId, name, step, values):
        """
        Put values into element block id and variable name at step.

        :type blockId: int
        :param blockId: The block id.
        :type name: str
        :param name: The name of the variable.
        :type step: int
        :param step: The time step at which to put the values.
        :type values: :class:`numpy.ndarray`
        :param values: The actual values.
        """
        assert step > 0, "Step must be larger than 0."
        # XXX: Currently the time axis is not unlimited due to a limitation
        # in h5netcdf - thus no new time steps can be created after the
        # initialization.
        assert step <= self._f.dimensions["time_step"]

        num_elem_name = "num_el_in_blk%i" % blockId
        assert num_elem_name in self._f.dimensions, \
            "Block id %i not found." % blockId

        # 1-based indexing!
        idx = self.get_element_variable_names().index(name) + 1

        variable_name = "vals_elem_var%ieb%i" % (idx, blockId)

        # If it does not exist, create it.
        if variable_name not in self._f.variables:
            self._f.create_variable(
                variable_name, ("time_step", num_elem_name),
                dtype=np.float64)

        self._f.variables[variable_name][step - 1] = values

    def set_node_variable_number(self, number):
        """
        Set number of node variables in exodus file.

        :type number: int
        :param number: The number of node variables.
        """
        if number == 0:  # pragma: no cover
            return

        self._f.dimensions["num_nod_var"] = number

        self._f.create_variable(
            "name_nod_var", ("num_nod_var", "len_name"),
            dtype="|S1")

        for _i in range(number):
            name = "vals_nod_var%i" % (_i + 1)
            self._f.create_variable(
                name, ("time_step", "num_nodes"),
                dtype=np.float64)

    def put_node_variable_name(self, name, index):
        """
        Node variable with name at index into exodus file.

        :type name: str
        :param name: The name of the element variable.
        :type index: int
        :param index: The index of the element variable. Starts with 1!
        """
        assert index < self._f.dimensions["num_nod_var"]

        self._f.variables["name_nod_var"][index - 1] = ""
        self._f.variables["name_nod_var"][index - 1, :len(name)] = \
            list(name)

    def get_node_variable_names(self):
        """
        Get list of node variable names in exodus file.
        """
        return ["".join(_i).strip()
                for _i in self._f.variables["name_nod_var"][:]]

    def put_node_variable_values(self, name, step, values):
        """
        Put node values into variable name at step into exodus file

        :type name: str
        :param name: The name of the variable.
        :type step: int
        :param step: The time step at which to put the values.
        :type values: :class:`numpy.ndarray`
        :param values: The actual values.
        """
        assert step > 0, "Step must be larger than 0."
        # XXX: Currently the time axis is not unlimited due to a limitation
        # in h5netcdf - thus no new time steps can be created after the
        # initialization.
        assert step <= self._f.dimensions["time_step"]

        # 1-based indexing!
        idx = self.get_node_variable_names().index(name) + 1

        d_name = "vals_nod_var%i" % idx
        self._f.variables[d_name][step - 1] = values

    def put_side_set_params(self, id, numSetSides, numSetDistFacts):
        """
        Set ID, num elements, and num nodes of a sideset

        :type id: int
        :param id: The id of the side set.
        :type numSetSides: int
        :param numSetSides: The number of elements for a side set.
        :type numSetDistFacts: int
        :param numSetDistFacts: The number of nodes for the side set.
        """
        assert numSetDistFacts == 0, "Only 0 is currently supported."

        max_ss = self._f.dimensions["num_side_sets"]

        cur_ss = [_i for _i in self._f.variables if _i.startswith("ss_prop")]
        ss_ids = [self._f.variables[_i][0] for _i in cur_ss]
        # -1 stands for now side set id.
        ss_ids = [_i for _i in ss_ids if _i >= 0]

        assert id not in ss_ids, "Side set id %i already exists." % id
        assert len(ss_ids) < max_ss, "Maximum number of side sets reached."

        idx = len(ss_ids) + 1
        dim_name = "num_side_ss%i" % idx
        elem_ss_name = "elem_ss%i" % idx
        side_ss_name = "side_ss%i" % idx

        # Create the dimension and variables.
        self._f.dimensions[dim_name] = numSetSides
        self._f.create_variable(elem_ss_name, (dim_name,), dtype=np.int32)
        self._f.create_variable(side_ss_name, (dim_name,), dtype=np.int32)

        # Set meta-data.
        self._f.variables["ss_status"][:] += 1
        self._f.variables["ss_prop%i" % idx][:] = id

    def put_side_set(self, id, sideSetElements, sideSetSides):
        """
        Set the id, element ids, and side ids for a side set (creates the
        side set).
        """
        # Find the side set.
        cur_ss = [_i for _i in self._f.variables if _i.startswith("ss_prop")]
        ss_ids = [_i for _i in cur_ss if self._f.variables[_i][0] == id]
        assert len(ss_ids) == 1, "Could not find side set with id %i." % id
        ss = ss_ids[0]
        idx = int(re.findall(r"\d+", ss)[0])
        elem_ss_name = "elem_ss%i" % idx
        side_ss_name = "side_ss%i" % idx

        self._f.variables[elem_ss_name][:] = sideSetElements
        self._f.variables[side_ss_name][:] = sideSetSides

    def put_side_set_name(self, id, name):
        """
        Write side set name for side set "id" in exodus file
        """
        # Find the side set.
        cur_ss = [_i for _i in self._f.variables if _i.startswith("ss_prop")]
        ss_ids = [_i for _i in cur_ss if self._f.variables[_i][0] == id]
        assert len(ss_ids) == 1, "Could not find side set with id %i." % id
        ss = ss_ids[0]

        # 1 based indexing.
        idx = int(re.findall(r"\d+", ss)[0]) - 1

        self._f.variables["ss_names"][idx - 1] = ""
        self._f.variables["ss_names"][idx - 1, :len(name)] = list(name)

    def _write_attrs(self, title):
        """
        Write all the attributes.
        """
        # XXX: Should probably all be defined in some header file.
        self._f.attrs['api_version'] = np.float32([6.30000019])
        self._f.attrs['version'] = np.float32([6.30000019])
        self._f.attrs['floating_point_word_size'] = \
            np.array([8], dtype=np.int32)
        self._f.attrs['file_size'] = np.array([1], dtype=np.int32)
        self._f.attrs['maximum_name_length'] = np.array([32],
                                                        dtype=np.int32)
        self._f.attrs['int64_status'] = np.array([0], dtype=np.int32)
        self._f.attrs['title'] = np.string_(title)

    def _create_variables(self):
        # Coordinate names.
        self._f.create_variable('/coor_names', ('num_dim', 'len_name'),
                                dtype='|S1')

        # Coordinates.
        _coords = "xyz"
        for _i in range(self._f.dimensions["num_dim"]):
            self._f.create_variable(
                '/coord' + _coords[_i], ('num_nodes',), dtype=np.float64)

        # Element block stuff.
        self._f.create_variable('/eb_names', ('num_el_blk', 'len_name'),
                                dtype='|S1')
        # I don't really understand the number here yet...
        self._f.create_variable('/eb_prop1', ('num_el_blk',),
                                dtype=np.int32, data=[-1])
        self._f.variables["eb_prop1"].attrs['name'] = np.string_('ID')
        self._f.create_variable('/eb_status', ('num_el_blk',),
                                dtype=np.int32)

        # Side sets.
        if "num_side_sets" in self._f.dimensions:
            self._f.create_variable(
                '/ss_names', ('num_side_sets', 'len_name'), dtype='|S1')
            self._f.create_variable(
                '/ss_prop1', ('num_side_sets',), dtype=np.int32, data=[-1])
            self._f.variables["ss_prop1"].attrs['name'] = np.string_('ID')
            self._f.create_variable('/ss_status', ('num_side_sets',),
                                    dtype=np.int32)

        # Time steps.
        self._f.create_variable('/time_whole', ('time_step',),
                                dtype=np.float64)

    def __del__(self):
        try:
            self._f.close()
        except:  # pragma: no cover
            pass

    def close(self):
        try:
            self._f.close()
        except:  # pragma: no cover
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
