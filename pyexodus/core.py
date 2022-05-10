#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (lionkrischer@gmail.com), 2022
:license:
    MIT License
"""
from __future__ import absolute_import

import os
import platform
import warnings

import numpy as np

import h5netcdf

from .compat import get_dim_size


# This uses zero based indexing to be compatible with numpy. The variables
# in the exodus files themselves are one based so keep that in mind!
# The values are from the exodus manual.
_SIDE_SET_NUMBERING = {
    "QUAD": np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32),
    "HEX": np.array(
        [
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [0, 4, 7, 3],
            [0, 3, 2, 1],
            [4, 5, 6, 7],
        ],
        dtype=np.int32,
    ),
}


class exodus(object):
    """
    Create a new Exodus file. Can also be used as a context manager.

    :type file: str
    :param file: Filename
    :type mode: str
    :param mode: File mode. Must currently be ``"r"``, ``"a"``, or ``"w"``.
    :type array_type: str
    :param array_type: Must be ``"numpy"``.
    :type title: str
    :param title: The title of the mesh.
    :type numDims: int
    :param numDims: The number of dimensions.
    :type numNodes: int
    :param numNodes: The number of nodes.
    :type numElems: int
    :param numElems: The number of elements.
    :type numBlocks: int
    :param numBlocks: The number of element blocks.
    :type numNodeSets: int
    :param numNodeSets: The number of node side sets.
    :type numSideSets: int
    :param numSideSets: The number of element side sets.
    :type io_size: int
    :param io_size: Determines how floating point variables are stored in
        the file.  Inputs will be converted if required.

        * ``0``: machine precision
        * ``4``: single precision
        * ``8``: double precision
    :type compression: tuple
    :param compression: Turn on compression. Pass a tuple of
        ``(method, option)``, e.g. ``("gzip", 2)``. Slows down writing a lot
        but the resulting files are potentially much smaller.
    """

    def __init__(
        self,
        file,
        mode="r",
        array_type="numpy",
        title=None,
        numDims=None,
        numNodes=None,
        numElems=None,
        numBlocks=None,
        numNodeSets=None,
        numSideSets=None,
        io_size=0,
        compression=None,
    ):

        if compression:
            self._comp_opts = {
                "chunks": True,
                "compression": compression[0],
                "compression_opts": compression[1],
            }
        else:
            self._comp_opts = {}

        # API is currently quite limited...mainly because nothing else is
        # implemented.
        assert mode in ["r", "a", "w"], "Only 'r', 'a', or 'w' is supported."
        assert array_type == "numpy", "array_type must be 'numpy'."

        if mode == "w":
            assert numNodeSets == 0, "numNodeSets must be 0 for now."
            assert numDims in [2, 3], "Only 2 or 3 dimensions are supported."

            # Determines the precision with which floating point variables are
            # written.
            if io_size == 0:
                if platform.architecture()[0] == "64bit":  # pragma: no cover
                    self.__f_dtype = np.float64
                    self.__f_word_size = 8
                else:  # pragma: no cover
                    self.__f_dtype = np.float32
                    self.__f_word_size = 4
            elif io_size == 4:
                self.__f_dtype = np.float32
                self.__f_word_size = 4
            elif io_size == 8:
                self.__f_dtype = np.float64
                self.__f_word_size = 8
            else:  # pragma: no cover
                raise NotImplementedError

            assert not os.path.exists(file), "File '%s' already exists." % file

            self._f = h5netcdf.File(file, mode="w")

            self._write_attrs(title=title)

            # Set the dimensions - very straightforward.
            # XXX: These should come from some header variable.
            self._f.dimensions["len_string"] = 33
            self._f.dimensions["len_line"] = 81
            # No clue what this is for...
            self._f.dimensions["four"] = 4
            self._f.dimensions["len_name"] = 256
            self._f.dimensions["time_step"] = None

            # These are dynamic.
            self._f.dimensions["num_dim"] = numDims
            self._f.dimensions["num_nodes"] = numNodes
            self._f.dimensions["num_elem"] = numElems
            self._f.dimensions["num_el_blk"] = numBlocks
            if numSideSets:
                self._f.dimensions["num_side_sets"] = numSideSets

            self._create_variables()

        elif mode in ["r", "a"]:
            if mode == "r":
                assert os.path.exists(file), "File '%s' does not exist." % file
            self._f = h5netcdf.File(file, mode=mode)

            # Currently no logic for this.
            if get_dim_size(self._f, "num_el_blk") > 1:  # pragma: no cover
                msg = (
                    "The file has more than one element block. pyexodus "
                    "currently contains no logic to deal with that. "
                    "Proceed at your own risk and best contact the "
                    "developers."
                )
                warnings.warn(msg)

        else:  # pragma: no cover
            raise NotImplementedError

    @property
    def num_dims(self):
        """
        Number of dimensions in the exodus file.
        """
        return int(get_dim_size(self._f, "num_dim"))

    def put_info_records(self, info):
        """
        Puts the info records into the exodus file.

        Does currently not do anything.

        :type info: list of str
        :param info: The strings to save.
        """
        if not info:
            return

        for _i, value in enumerate(info):
            assert len(value) < get_dim_size(
                self._f, "len_line"
            ), "Records '%s' is longer then %i letters." % (
                value,
                get_dim_size(self._f, "len_line"),
            )

        self._f.dimensions["num_info"] = len(info)

        self._f.create_variable(
            "info_records",
            ("num_info", "len_line"),
            dtype="|S1",
            **self._comp_opts
        )

        ir = self._f.variables["info_records"]
        for idx, value in enumerate(info):
            # Clear just to be safe.
            ir[idx] = b""
            if not value:
                continue
            ir[idx, : len(value)] = [
                _i.encode() if hasattr(_i, "encode") else _i for _i in value
            ]

    def put_coords(self, xCoords, yCoords, zCoords):
        """
        Put coordinates in the exodus file.

        :type xCoords: :class:`numpy.ndarray`
        :param xCoords: The X coordiantes.
        :type yCoords: :class:`numpy.ndarray`
        :param yCoords:  The Y coordinates.
        :type zCoords: :class:`numpy.ndarray`
        :param zCoords:  The Z coordinates.
        """
        self._f.variables["coordx"][:] = xCoords
        self._f.variables["coordy"][:] = yCoords
        self._f.variables["coordz"][:] = zCoords

    def put_elem_blk_info(
        self, id, elemType, numElems, numNodesPerElem, numAttrsPerElem
    ):
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
        assert numElems <= get_dim_size(
            self._f, "num_elem"
        ), "Canont have more elements in the block then globally set."
        assert numAttrsPerElem == 0, "Must be 0 for now."

        # So the logic is as follows. `eb_status` keeps track of which
        # element ids have already been assigned. We find the first that is
        # not zero and that is the actual index of the the element block.
        status = self._f._variables["eb_status"][:]
        assert 0 in status, "All element blocks already set."
        idx = np.argwhere(status == 0)[0][0] + 1

        num_el_name = "num_el_in_blk%i" % idx
        num_node_per_el_name = "num_nod_per_el%i" % idx
        var_name = "connect%i" % idx

        self._f.dimensions[num_el_name] = numElems
        self._f.dimensions[num_node_per_el_name] = numNodesPerElem

        self._f.create_variable(
            var_name,
            (num_el_name, num_node_per_el_name),
            dtype=np.int32,
            **self._comp_opts
        )
        self._f.variables[var_name].attrs["elem_type"] = np.string_(elemType)

        # Set the status and thus "claim" the element block id.
        self._f.variables["eb_status"][idx - 1] = 1
        # For some reason this is always eb_prop1.
        self._f.variables["eb_prop1"][idx - 1] = id

    def put_elem_connectivity(
        self, id, connectivity, shift_indices=0, chunk_size_in_mb=128
    ):
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
        :type shift_indices: int
        :param shift_indices: **Not available in the official exodus Python
            API!** This value will be added to all indices before they are
            written to the file. This is useful if you for example
            internally work with a 0-based indexing scheme but exodus
            requires a 1-based indexing scheme.
        :type chunk_size_in_mb: int
        :param chunk_size_in_mb: If ``shift_indices`` != 0 values will be
            written in chunks of this size. This is also the maximum memory
            usage of this method. If ``shift_indices`` == 0, all indices
            will be written directly from memory and no additional memory is
            required.
        """
        num_el_name = "num_el_in_blk%i" % id
        num_node_per_el_name = "num_nod_per_el%i" % id
        var_name = "connect%i" % id

        assert num_el_name in self._f.dimensions, (
            "Block id %i does not exist" % id
        )
        assert num_node_per_el_name in self._f.dimensions, (
            "Block id %i does not exist" % id
        )

        assert connectivity.size == (
            get_dim_size(self._f, num_el_name)
            * get_dim_size(self._f, num_node_per_el_name)
        )

        if shift_indices:
            ne = get_dim_size(self._f, num_el_name)
            nn = get_dim_size(self._f, num_node_per_el_name)

            chunk_size = int(
                chunk_size_in_mb * 1024**2 / connectivity.dtype.itemsize / nn
            )

            _t = connectivity.reshape((ne, nn))

            idx = 0
            while idx < ne:
                self._f.variables[var_name][idx : idx + chunk_size] = (  # NOQA
                    _t[idx : idx + chunk_size] + shift_indices  # NOQA
                )
                idx += chunk_size
        else:
            self._f.variables[var_name][:] = connectivity.reshape(
                (
                    get_dim_size(self._f, num_el_name),
                    get_dim_size(self._f, num_node_per_el_name),
                )
            )

    def put_time(self, step, value):
        """
        Put time step and value into exodus file.

        :type step: int
        :param step: The index of the time step. First is 1.
        :type value: float
        :param value: The actual time at that index.
        """
        self.__resize_time_if_necessary(step)
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
            "name_glo_var",
            ("num_glo_var", "len_name"),
            dtype="|S1",
            **self._comp_opts
        )
        self._f.create_variable(
            "vals_glo_var",
            ("time_step", "num_glo_var"),
            dtype=self.__f_dtype,
            **self._comp_opts
        )

    def put_global_variable_name(self, name, index):
        """
        Put global variable with name at index into exodus file.

        :type name: str
        :param name: The name of the variable.
        :type index: int
        :param index: The index of the global variable. First is 1!
        """
        self._f.variables["name_glo_var"][index - 1] = b""
        self._f.variables["name_glo_var"][index - 1, : len(name)] = [
            _i.encode() if hasattr(_i, "encode") else _i for _i in name
        ]

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
        self.__resize_time_if_necessary(step)
        idx = self.get_global_variable_names().index(name)
        self._f.variables["vals_glo_var"][step - 1, idx] = value

    def get_global_variable_names(self):
        """
        Get list of global variable names in exodus file.
        """
        return [
            b"".join(_i).strip().decode()
            for _i in self._f.variables["name_glo_var"][:]
        ]

    def get_global_variable_values(self, name):
        """
        Get global variables values within an exodus file.
        """
        idx = self.get_global_variable_names().index(name)
        return self._f.variables["vals_glo_var"][0, idx]

    def set_element_variable_number(self, number):
        """
        Set number of element variables in exodus file.

        :type number: int
        :param number: The number of variables per element.
        """
        if not number:
            return

        self._f.dimensions["num_elem_var"] = number

        self._f.create_variable(
            "name_elem_var",
            ("num_elem_var", "len_name"),
            dtype="|S1",
            **self._comp_opts
        )

    def put_element_variable_name(self, name, index):
        """
        Element variable with name at index into exodus file.

        :type name: str
        :param name: The name of the element variable.
        :type index: int
        :param index: The index of the element variable. Starts with 1!
        """
        self._f.variables["name_elem_var"][index - 1] = b""
        self._f.variables["name_elem_var"][index - 1, : len(name)] = [
            _i.encode() if hasattr(_i, "encode") else _i for _i in name
        ]

    def get_element_variable_names(self):
        """
        Get list of element variable names in exodus file.
        """
        return [
            b"".join(_i).strip().decode()
            for _i in self._f.variables["name_elem_var"][:]
        ]

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
        self.__resize_time_if_necessary(step)

        num_elem_name = "num_el_in_blk%i" % blockId
        assert num_elem_name in self._f.dimensions, (
            "Block id %i not found." % blockId
        )

        # 1-based indexing!
        idx = self.get_element_variable_names().index(name) + 1

        variable_name = "vals_elem_var%ieb%i" % (idx, blockId)

        # If it does not exist, create it.
        if variable_name not in self._f.variables:
            self._f.create_variable(
                variable_name,
                ("time_step", num_elem_name),
                dtype=self.__f_dtype,
                **self._comp_opts
            )

        self._f.variables[variable_name][step - 1] = values

    def get_element_variable_values(self, blockId, name, step):
        """
        Get values from element block id and variable name at step.

        :type blockId: int
        :param blockId: The block id.
        :type name: str
        :param name: The name of the variable.
        :type step: int
        :param step: The time step at which to put the values.
        Return values: The actual values.
        """
        assert step > 0, "Step must be larger than 0."
        assert get_dim_size(
            self._f, "time_step"
        ) is None or step <= get_dim_size(self._f, "time_step")

        num_elem_name = "num_el_in_blk%i" % blockId
        assert num_elem_name in self._f.dimensions, (
            "Block id %i not found." % blockId
        )

        # 1-based indexing!
        idx = self.get_element_variable_names().index(name) + 1

        variable_name = "vals_elem_var%ieb%i" % (idx, blockId)

        # If it does not exist, raise exception
        assert variable_name in self._f.variables, (
            "Variable %s not found" % variable_name
        )

        return self._f.variables[variable_name][step - 1][:]

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
            "name_nod_var",
            ("num_nod_var", "len_name"),
            dtype="|S1",
            **self._comp_opts
        )

        for _i in range(number):
            name = "vals_nod_var%i" % (_i + 1)
            self._f.create_variable(
                name,
                ("time_step", "num_nodes"),
                dtype=self.__f_dtype,
                **self._comp_opts
            )

    def put_node_variable_name(self, name, index):
        """
        Node variable with name at index into exodus file.

        :type name: str
        :param name: The name of the element variable.
        :type index: int
        :param index: The index of the element variable. Starts with 1!
        """
        # 1 - based indexing!
        assert index <= get_dim_size(self._f, "num_nod_var")

        self._f.variables["name_nod_var"][index - 1] = b""
        self._f.variables["name_nod_var"][index - 1, : len(name)] = [
            _i.encode() if hasattr(_i, "encode") else _i for _i in name
        ]

    def get_node_variable_names(self):
        """
        Get list of node variable names in exodus file.
        """
        return [
            b"".join(_i).strip().decode()
            for _i in self._f.variables["name_nod_var"][:]
        ]

    def get_node_variable_number(self):
        """
        Get number of node variables in the file.
        """
        if "num_nod_var" not in self._f.dimensions:
            return 0
        return int(get_dim_size(self._f, "num_nod_var"))

    def __resize_time_if_necessary(self, step):
        assert step > 0, "Step must be larger than 0."
        d = self._f.dimensions["time_step"]
        # Compatibility.
        if isinstance(d, (int, np.integer)) or d is None:
            assert d is None or step <= d
            if step > self._f._current_dim_sizes["time_step"]:
                self._f.resize_dimension("time_step", step)
        else:
            assert d is None or d.isunlimited()
            if step > d.size:
                self._f.resize_dimension("time_step", step)

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
        self.__resize_time_if_necessary(step)

        # 1-based indexing!
        idx = self.get_node_variable_names().index(name) + 1

        d_name = "vals_nod_var%i" % idx
        self._f.variables[d_name][step - 1] = values

    def get_node_variable_values(self, name, step):
        """
        Get the node variable values for a a certain step.

        :type name: str
        :param name: The name of the variable.
        :type step: int
        :param step: The time step at which to get the values.
        """
        # Make sure the step is valid.
        if step <= 0:
            raise ValueError("Step must be larger than zero.")
        # This cannot really happen anymore with newer pyexodus versions, as
        # the time_step is always an unlimited dimension so there is not easy
        # way to test this. But it might still happen if somebody reads a file
        # created with an older pyexodus version so I'll leave the block in
        # but mark it as uncovered by the tests.
        if get_dim_size(self._f, "time_step") is not None and not (
            0 < step <= get_dim_size(self._f, "time_step")
        ):  # pragma: no cover
            msg = "Step must be 0 < step < %i." % get_dim_size(
                self._f, "time_step"
            )
            raise ValueError(msg)
        # Will raise with a reasonable error message if name is not correct.
        # 1-based indexing!
        idx = self.get_node_variable_names().index(name) + 1

        d_name = "vals_nod_var%i" % idx
        # If it is resizeable, check the actual size.
        if get_dim_size(self._f, "time_step") is None:
            available_steps = self._f.variables[d_name].shape[0]
            if not (0 < step <= available_steps):
                msg = "Step must be 0 < step <= %i." % available_steps
                raise ValueError(msg)

        return self._f.variables[d_name][step - 1][:]

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

        assert id not in self._f.variables["ss_prop1"][:], (
            "Side set id %i already exists." % id
        )

        _t = self._f.variables["ss_status"][:]
        count = len(_t[_t > 0])
        assert count < get_dim_size(
            self._f, "num_side_sets"
        ), "Maximum number of side sets reached."

        idx = count + 1
        dim_name = "num_side_ss%i" % idx
        elem_ss_name = "elem_ss%i" % idx
        side_ss_name = "side_ss%i" % idx

        # Create the dimension and variables.
        self._f.dimensions[dim_name] = numSetSides
        self._f.create_variable(
            elem_ss_name, (dim_name,), dtype=np.int32, **self._comp_opts
        )
        self._f.create_variable(
            side_ss_name, (dim_name,), dtype=np.int32, **self._comp_opts
        )

        # Set meta-data.
        self._f.variables["ss_status"][idx - 1] = 1
        # For reasons I don't understand, this is ALWAYS ss_prop1.
        self._f.variables["ss_prop1"][idx - 1] = id

    def put_side_set(self, id, sideSetElements, sideSetSides):
        """
        Set the id, element ids, and side ids for a side set (creates the
        side set).

        :type id: int
        :param id: The id of the side set.
        :type sideSetElements: :class:`numpy.ndarray`
        :param sideSetElements: The side set elements.
        :type sideSetSides: :class:`numpy.ndarray`
        :param sideSetSides: The side set sides.
        """
        # Find the side set.
        _idx = self._f.variables["ss_prop1"][:]
        assert id in _idx, "Could not find side set with id %i." % id
        # 1-based indexing!
        idx = np.argwhere(_idx == id)[0][0] + 1

        elem_ss_name = "elem_ss%i" % idx
        side_ss_name = "side_ss%i" % idx

        self._f.variables[elem_ss_name][:] = sideSetElements
        self._f.variables[side_ss_name][:] = sideSetSides

    def put_side_set_name(self, id, name):
        """
        Write side set name for side set "id" in exodus file

        :type id: int
        :param id: The id of the side set.
        :type name: str
        :param name: The string of the side set.
        """
        # Find the side set.
        _idx = self._f.variables["ss_prop1"][:]
        assert id in _idx, "Could not find side set with id %i." % id
        # 1-based indexing!
        idx = np.argwhere(_idx == id)[0][0] + 1

        self._f.variables["ss_names"][idx - 1] = b""
        self._f.variables["ss_names"][idx - 1, : len(name)] = [
            _i.encode() if hasattr(_i, "encode") else _i for _i in name
        ]

    def get_side_set_names(self):
        """
        Get a list of the side set names in the exodus file.
        """
        _side_sets = self._f.variables["ss_names"][:]
        side_sets = []
        for _i in _side_sets:
            side_sets.append(
                "".join(
                    _j.decode() if hasattr(_j, "decode") else _j for _j in _i
                )
            )
        return side_sets

    def get_side_set_ids(self):
        """
        Get a list of side set ids in the exodus file.
        """
        return [int(_i) for _i in self._f.variables["ss_prop1"][:]]

    def get_side_set(self, id):
        """
        Get element and side ids for a certain side set.

        Returns a tuple of two arrays. The first are the element indices,
        the second the side ids in these elements.

        :type id: int
        :param id: The id of the side set.
        """
        ids = self.get_side_set_ids()
        if id not in ids:
            raise ValueError(
                "No side set with id %i in file. Available "
                "ids: %s." % (id, ", ".join(["%i" % _i for _i in ids]))
            )
        id = ids.index(id) + 1
        side_name = "side_ss%i" % id
        elem_name = "elem_ss%i" % id

        return self._f.variables[elem_name][:], self._f.variables[side_name][:]

    def get_elem_type_for_block(self, id):
        """
        Return the element type for an element block as a string.

        .. note::

            This method does not have a counter part in the official exodus
            Python API.

        :type id: int
        :param id: The element block id.
        """
        var_name = "connect%i" % id
        if var_name not in self._f.variables:
            raise ValueError("No element block with id %i in file." % id)
        elem_type = self._f[var_name].attrs["elem_type"]
        try:
            elem_type = elem_type.decode()
        # Not triggered on all Python versions.
        except AttributeError:  # pragma: no cover
            pass
        return elem_type

    def get_side_set_node_list(self, id):
        """
        Get the nodes for a certain side set.

        Returns a tuple of two arrays. The first contains the number of
        nodes on each face the the seconds are the local node ids for the
        side set in the exodus file.

        :type id: int
        :param id: The id of the side set.
        """
        # XXX: Currently only works for files with a single element block.
        elem_type = self.get_elem_type_for_block(id=1)

        elem_idx, side_idx = self.get_side_set(id=id)
        _sin = _SIDE_SET_NUMBERING[elem_type]

        num_nodes = np.ones_like(elem_idx) * _sin.shape[1]
        # This one is a bit tricky. Not sure if the current solution is
        # optimal but it gets the trick done and does avoid a bunch of copies.
        # Step 1: Get all elements in the side set. Account for 1-based
        # indexing
        _e = self._f.variables["connect1"][:][elem_idx - 1]
        # Step 2: From each element we have to pick these sides.
        _s = _sin[side_idx - 1]
        # This still has some additional allocations but otherwise indexes
        # pretty directly.
        local_node_ids = _e.ravel()[
            (_s + np.arange(_e.shape[0])[:, np.newaxis] * _e.shape[1]).ravel()
        ]

        return num_nodes, local_node_ids

    def get_coord(self, i):
        """
        Get x, y, z of i-th node in the exodus file.

        :type i: int or array/list of ints
        :param i: Node index or indices. 1-based. The official exodus API
            can only take single indices - pyexodus can take a list of indices.
            In that case it will still return a three-tuple, but each now
            contains multiple variables.
        """
        # Make it work with single indices and arrays.
        i = list(np.atleast_1d(i) - 1)
        if len(i) == 1:
            i = i[0]
            if not 1 <= i + 1 <= get_dim_size(self._f, "num_nodes"):
                raise ValueError(
                    "Invalid index. Coordinate bounds: [1, %i]."
                    % get_dim_size(self._f, "num_nodes")
                )

        x = self._f.variables["coordx"][i]
        y = self._f.variables["coordy"][i]
        if get_dim_size(self._f, "num_dim") == 2:
            return x, y, np.zeros_like(x)
        return x, y, self._f.variables["coordz"][i]

    def get_coords(self):
        """
        Returns all nodes in x, y, z.
        """
        x = self._f.variables["coordx"][:]
        y = self._f.variables["coordy"][:]
        if get_dim_size(self._f, "num_dim") == 2:
            return x, y, np.zeros_like(x)
        return x, y, self._f.variables["coordz"][:]

    def get_elem_connectivity(self, id, indices=None):
        """
        Get the connectivity for a certain element block.

        Returns a tuple with three things: The actual connectivity as an
        array of node indices, the number of elements in this block, and the
        number of nodes per elements in this block.

        :type id: int
        :param id: Id of the element block.
        :type indices: :class:`numpy.ndarray`
        :param indices: If given, only get the connectivity of the specified
            element indices relative to the element block. These are 1-based
            indices in accordance with the exodus convention! Note that the
            second and third returned item are always stats for the whole
            connectivity array regardless of this argument. This parameter
            is not part of the official exodus Python API.
        """
        var_name = "connect%i" % id
        conn = self._f.variables[var_name]

        # Read everything if indices is not given.
        if indices is None:
            indices = slice(None)
        else:
            indices = np.array(indices)
            indices = list(indices - 1)

        return conn[indices], conn.shape[0], conn.shape[1]

    def _write_attrs(self, title):
        """
        Write all the attributes.
        """
        self._f.attrs["api_version"] = np.float32([7.05])
        self._f.attrs["version"] = np.float32([7.05])
        self._f.attrs["floating_point_word_size"] = np.array(
            [self.__f_word_size], dtype=np.int32
        )
        self._f.attrs["file_size"] = np.array([1], dtype=np.int32)
        self._f.attrs["maximum_name_length"] = np.array([32], dtype=np.int32)
        self._f.attrs["int64_status"] = np.array([0], dtype=np.int32)
        self._f.attrs["title"] = np.string_(title)

    def _create_variables(self):
        # Time steps.
        self._f.create_variable(
            "/time_whole",
            ("time_step",),
            dtype=self.__f_dtype,
            **self._comp_opts
        )

        # Element block stuff.
        self._f.create_variable(
            "/eb_names",
            ("num_el_blk", "len_name"),
            dtype="|S1",
            **self._comp_opts
        )
        self._f.create_variable(
            "/eb_status", ("num_el_blk",), dtype=np.int32, **self._comp_opts
        )
        # I don't really understand the number here yet...
        self._f.create_variable(
            "/eb_prop1",
            ("num_el_blk",),
            dtype=np.int32,
            data=[-1] * get_dim_size(self._f, "num_el_blk"),
            **self._comp_opts
        )
        self._f.variables["eb_prop1"].attrs["name"] = np.string_("ID")

        # Coordinate names.
        self._f.create_variable(
            "/coor_names",
            ("num_dim", "len_name"),
            dtype="|S1",
            **self._comp_opts
        )

        # Coordinates.
        for i in "xyz":
            self._f.create_variable(
                "/coord" + i,
                ("num_nodes",),
                dtype=self.__f_dtype,
                **self._comp_opts
            )

        # Side sets.
        if "num_side_sets" in self._f.dimensions:
            self._f.create_variable(
                "/ss_names",
                ("num_side_sets", "len_name"),
                dtype="|S1",
                **self._comp_opts
            )
            self._f.create_variable(
                "/ss_prop1",
                ("num_side_sets",),
                dtype=np.int32,
                data=[-1] * get_dim_size(self._f, "num_side_sets"),
                **self._comp_opts
            )
            self._f.variables["ss_prop1"].attrs["name"] = np.string_("ID")
            self._f.create_variable(
                "/ss_status",
                ("num_side_sets",),
                dtype=np.int32,
                **self._comp_opts
            )

    def __del__(self):
        try:
            self._f.close()
        except Exception:  # pragma: no cover
            pass

    def close(self):
        try:
            self._f.close()
        except Exception:  # pragma: no cover
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
