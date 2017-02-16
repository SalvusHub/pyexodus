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
    _p.append({"io_size": 0, "word_size": 8, "f_dtype": np.float64},)


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
