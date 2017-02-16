#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (lionkrischer@gmail.com), 2017
:license:
    MIT License
"""
import os
import platform

import numpy as np
import pytest


from pyexodus import exodus


_p = [
    {"io_size": 4, "word_size": 4, "f_dtype": np.float32},
    {"io_size": 8, "word_size": 8, "f_dtype": np.float64},
]

if platform.architecture()[0] == "64bit":  # pragma: no cover
    _p.append({"io_size": 0, "word_size": 8, "f_dtype": np.float64},)
else:  # pragma: no cover
    _p.append({"io_size": 0, "word_size": 4, "f_dtype": np.float32},)


@pytest.fixture(params=_p, ids=["io_size_%i" % _i["io_size"] for _i in _p])
def io_size(request):
    """
    Fixture to parametrize over the io_sizes.
    """
    return request.param


def test_get_side_set_names(tmpdir, io_size):
    filename = os.path.join(tmpdir.strpath, "example.e")

    # Put two side sets. Note that the ids of the side sets have little to
    # do with their naming inside the file...
    with exodus(filename, mode="w", title="Example", array_type="numpy",
                numDims=3, numNodes=5, numElems=6, numBlocks=1,
                numNodeSets=0, numSideSets=2, io_size=io_size["io_size"]) as e:
        e.put_side_set_params(4, 5, 0)
        e.put_side_set(4, np.ones(5, dtype=np.int32) * 2,
                       np.ones(5, dtype=np.int32) * 3)
        e.put_side_set_name(4, "edge of the world")

        e.put_side_set_params(2, 5, 0)
        e.put_side_set(2, np.ones(5, dtype=np.int32) * 7,
                       np.ones(5, dtype=np.int32) * 8)
        e.put_side_set_name(2, "hallo")

    with exodus(filename, mode="r") as e:
        assert e.get_side_set_names() == ["edge of the world", "hallo"]


def test_get_side_set_ids(tmpdir, io_size):
    filename = os.path.join(tmpdir.strpath, "example.e")

    # Put two side sets. Note that the ids of the side sets have little to
    # do with their naming inside the file...
    with exodus(filename, mode="w", title="Example", array_type="numpy",
                numDims=3, numNodes=5, numElems=6, numBlocks=1,
                numNodeSets=0, numSideSets=2, io_size=io_size["io_size"]) as e:
        e.put_side_set_params(4, 5, 0)
        e.put_side_set(4, np.ones(5, dtype=np.int32) * 2,
                       np.ones(5, dtype=np.int32) * 3)
        e.put_side_set_name(4, "edge of the world")

        e.put_side_set_params(2, 5, 0)
        e.put_side_set(2, np.ones(5, dtype=np.int32) * 7,
                       np.ones(5, dtype=np.int32) * 8)
        e.put_side_set_name(2, "hallo")

    with exodus(filename, mode="r") as e:
        assert e.get_side_set_ids() == [4, 2]


def test_get_side_set(tmpdir, io_size):
    filename = os.path.join(tmpdir.strpath, "example.e")

    # Hex elements.
    with exodus(filename, mode="w", title="Example", array_type="numpy",
                numDims=3, numNodes=5, numElems=6, numBlocks=1,
                numNodeSets=0, numSideSets=1, io_size=io_size["io_size"]) as e:
        e.put_coords(
            xCoords=np.arange(5, dtype=np.float64),
            yCoords=np.arange(5, dtype=np.float64) * 2,
            zCoords=np.arange(5, dtype=np.float64) * 3
        )
        e.put_elem_blk_info(1, "HEX", 6, 8, 0)
        e.put_elem_connectivity(1, np.arange(6 * 8) + 7)
        e.put_side_set_params(4, 5, 0)
        e.put_side_set(4, np.arange(5, dtype=np.int32) + 2,
                       np.arange(5, dtype=np.int32) + 1)

    with exodus(filename, mode="r") as e:
        elem_idx, side_ids = e.get_side_set(id=4)

    np.testing.assert_equal(elem_idx, [2, 3, 4, 5, 6])
    np.testing.assert_equal(side_ids, [1, 2, 3, 4, 5])


def test_get_side_set_node_list(tmpdir, io_size):
    filename = os.path.join(tmpdir.strpath, "example.e")

    # Hex elements.
    with exodus(filename, mode="w", title="Example", array_type="numpy",
                numDims=3, numNodes=5, numElems=6, numBlocks=1,
                numNodeSets=0, numSideSets=1, io_size=io_size["io_size"]) as e:
        e.put_coords(
            xCoords=np.arange(5, dtype=np.float64),
            yCoords=np.arange(5, dtype=np.float64) * 2,
            zCoords=np.arange(5, dtype=np.float64) * 3
        )
        e.put_elem_blk_info(1, "HEX", 6, 8, 0)
        e.put_elem_connectivity(1, np.arange(6 * 8) + 7)
        e.put_side_set_params(4, 5, 0)
        e.put_side_set(4, np.arange(5, dtype=np.int32) + 2,
                       np.arange(5, dtype=np.int32) + 1)

    with exodus(filename, mode="r") as e:
        num_nodes, local_node_ids = e.get_side_set_node_list(id=4)

    np.testing.assert_equal(num_nodes, [4, 4, 4, 4, 4])
    np.testing.assert_equal(
        local_node_ids,
        [15, 16, 20, 19, 24, 25, 29, 28, 33, 34, 38, 37, 39, 43, 46, 42, 47,
         50, 49, 48])
