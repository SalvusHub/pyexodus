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
    _p.append(
        {"io_size": 0, "word_size": 8, "f_dtype": np.float64},
    )
else:  # pragma: no cover
    _p.append(
        {"io_size": 0, "word_size": 4, "f_dtype": np.float32},
    )


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
    with exodus(
        filename,
        mode="w",
        title="Example",
        array_type="numpy",
        numDims=3,
        numNodes=5,
        numElems=6,
        numBlocks=1,
        numNodeSets=0,
        numSideSets=2,
        io_size=io_size["io_size"],
    ) as e:
        e.put_side_set_params(4, 5, 0)
        e.put_side_set(
            4, np.ones(5, dtype=np.int32) * 2, np.ones(5, dtype=np.int32) * 3
        )
        e.put_side_set_name(4, "edge of the world")

        e.put_side_set_params(2, 5, 0)
        e.put_side_set(
            2, np.ones(5, dtype=np.int32) * 7, np.ones(5, dtype=np.int32) * 8
        )
        e.put_side_set_name(2, "hallo")

    with exodus(filename, mode="r") as e:
        assert e.get_side_set_names() == ["edge of the world", "hallo"]


def test_read_global_variables(tmpdir, io_size):

    filename = os.path.join(tmpdir.strpath, "example.e")
    with exodus(
        filename,
        mode="w",
        title="Example",
        array_type="numpy",
        numDims=3,
        numNodes=5,
        numElems=6,
        numBlocks=1,
        numNodeSets=0,
        numSideSets=2,
        io_size=io_size["io_size"],
    ) as e:

        e.set_global_variable_number(1)
        e.put_global_variable_name("Test", 1)
        e.put_global_variable_value("Test", 1, 1.1)

    with (exodus(filename, mode="r")) as e:
        assert e.get_global_variable_names() == ["Test"]
        value = e.get_global_variable_values("Test")
        np.testing.assert_almost_equal(value, 1.1)


def test_get_side_set_ids(tmpdir, io_size):
    filename = os.path.join(tmpdir.strpath, "example.e")

    # Put two side sets. Note that the ids of the side sets have little to
    # do with their naming inside the file...
    with exodus(
        filename,
        mode="w",
        title="Example",
        array_type="numpy",
        numDims=3,
        numNodes=5,
        numElems=6,
        numBlocks=1,
        numNodeSets=0,
        numSideSets=2,
        io_size=io_size["io_size"],
    ) as e:
        e.put_side_set_params(4, 5, 0)
        e.put_side_set(
            4, np.ones(5, dtype=np.int32) * 2, np.ones(5, dtype=np.int32) * 3
        )
        e.put_side_set_name(4, "edge of the world")

        e.put_side_set_params(2, 5, 0)
        e.put_side_set(
            2, np.ones(5, dtype=np.int32) * 7, np.ones(5, dtype=np.int32) * 8
        )
        e.put_side_set_name(2, "hallo")

    with exodus(filename, mode="r") as e:
        assert e.get_side_set_ids() == [4, 2]


def test_get_side_set_hex(tmpdir, io_size):
    filename = os.path.join(tmpdir.strpath, "example.e")

    # Hex elements.
    with exodus(
        filename,
        mode="w",
        title="Example",
        array_type="numpy",
        numDims=3,
        numNodes=5,
        numElems=6,
        numBlocks=1,
        numNodeSets=0,
        numSideSets=1,
        io_size=io_size["io_size"],
    ) as e:
        e.put_coords(
            xCoords=np.arange(5, dtype=np.float64),
            yCoords=np.arange(5, dtype=np.float64) * 2,
            zCoords=np.arange(5, dtype=np.float64) * 3,
        )
        e.put_elem_blk_info(1, "HEX", 6, 8, 0)
        e.put_elem_connectivity(1, np.arange(6 * 8) + 7)
        e.put_side_set_params(4, 5, 0)
        e.put_side_set(
            4,
            np.arange(5, dtype=np.int32) + 2,
            np.arange(5, dtype=np.int32) + 1,
        )

    with exodus(filename, mode="r") as e:
        elem_idx, side_ids = e.get_side_set(id=4)

    np.testing.assert_equal(elem_idx, [2, 3, 4, 5, 6])
    np.testing.assert_equal(side_ids, [1, 2, 3, 4, 5])


def test_get_side_set_quad(tmpdir, io_size):
    filename = os.path.join(tmpdir.strpath, "example.e")

    with exodus(
        filename,
        mode="w",
        title="Example",
        array_type="numpy",
        numDims=2,
        numNodes=5,
        numElems=6,
        numBlocks=1,
        numNodeSets=0,
        numSideSets=1,
        io_size=io_size["io_size"],
    ) as e:
        e.put_coords(
            xCoords=np.arange(5, dtype=np.float64),
            yCoords=np.arange(5, dtype=np.float64) * 2,
            zCoords=np.zeros(5),
        )
        e.put_elem_blk_info(1, b"QUAD", 6, 4, 0)
        e.put_elem_connectivity(1, np.arange(6 * 4) + 7)
        e.put_side_set_params(4, 4, 0)
        e.put_side_set(
            4,
            np.arange(4, dtype=np.int32) + 2,
            np.arange(4, dtype=np.int32) + 1,
        )

    with exodus(filename, mode="r") as e:
        elem_idx, side_ids = e.get_side_set(id=4)

    np.testing.assert_equal(elem_idx, [2, 3, 4, 5])
    np.testing.assert_equal(side_ids, [1, 2, 3, 4])


def test_get_side_set_invalid_id(tmpdir, io_size):
    filename = os.path.join(tmpdir.strpath, "example.e")

    # Put two side sets. Note that the ids of the side sets have little to
    # do with their naming inside the file...
    with exodus(
        filename,
        mode="w",
        title="Example",
        array_type="numpy",
        numDims=3,
        numNodes=5,
        numElems=6,
        numBlocks=1,
        numNodeSets=0,
        numSideSets=2,
        io_size=io_size["io_size"],
    ) as e:
        e.put_side_set_params(4, 5, 0)
        e.put_side_set(
            4, np.ones(5, dtype=np.int32) * 2, np.ones(5, dtype=np.int32) * 3
        )
        e.put_side_set_name(4, "edge of the world")

        e.put_side_set_params(2, 5, 0)
        e.put_side_set(
            2, np.ones(5, dtype=np.int32) * 7, np.ones(5, dtype=np.int32) * 8
        )
        e.put_side_set_name(2, "hallo")

    with exodus(filename, mode="r") as e:
        with pytest.raises(ValueError) as f:
            e.get_side_set(id=7)

    assert (
        f.value.args[0]
        == "No side set with id 7 in file. Available ids: 4, 2."
    )


def test_get_side_set_node_list_hex(tmpdir, io_size):
    filename = os.path.join(tmpdir.strpath, "example.e")

    # Hex elements.
    with exodus(
        filename,
        mode="w",
        title="Example",
        array_type="numpy",
        numDims=3,
        numNodes=5,
        numElems=6,
        numBlocks=1,
        numNodeSets=0,
        numSideSets=1,
        io_size=io_size["io_size"],
    ) as e:
        e.put_coords(
            xCoords=np.arange(5, dtype=np.float64),
            yCoords=np.arange(5, dtype=np.float64) * 2,
            zCoords=np.arange(5, dtype=np.float64) * 3,
        )
        e.put_elem_blk_info(1, "HEX", 6, 8, 0)
        e.put_elem_connectivity(1, np.arange(6 * 8) + 7)
        e.put_side_set_params(4, 5, 0)
        e.put_side_set(
            4,
            np.arange(5, dtype=np.int32) + 2,
            np.arange(5, dtype=np.int32) + 1,
        )

    with exodus(filename, mode="r") as e:
        num_nodes, local_node_ids = e.get_side_set_node_list(id=4)

    np.testing.assert_equal(num_nodes, [4, 4, 4, 4, 4])
    np.testing.assert_equal(
        local_node_ids,
        [
            15,
            16,
            20,
            19,
            24,
            25,
            29,
            28,
            33,
            34,
            38,
            37,
            39,
            43,
            46,
            42,
            47,
            50,
            49,
            48,
        ],
    )


def test_get_side_set_node_list_quad(tmpdir, io_size):
    filename = os.path.join(tmpdir.strpath, "example.e")

    with exodus(
        filename,
        mode="w",
        title="Example",
        array_type="numpy",
        numDims=2,
        numNodes=5,
        numElems=6,
        numBlocks=1,
        numNodeSets=0,
        numSideSets=1,
        io_size=io_size["io_size"],
    ) as e:
        e.put_coords(
            xCoords=np.arange(5, dtype=np.float64),
            yCoords=np.arange(5, dtype=np.float64) * 2,
            zCoords=np.zeros(5),
        )
        e.put_elem_blk_info(1, b"QUAD", 6, 4, 0)
        e.put_elem_connectivity(1, np.arange(6 * 4) + 7)
        e.put_side_set_params(4, 4, 0)
        e.put_side_set(
            4,
            np.arange(4, dtype=np.int32) + 2,
            np.arange(4, dtype=np.int32) + 1,
        )

    with exodus(filename, mode="r") as e:
        num_nodes, local_node_ids = e.get_side_set_node_list(id=4)

    np.testing.assert_equal(num_nodes, [2, 2, 2, 2])
    np.testing.assert_equal(local_node_ids, [11, 12, 16, 17, 21, 22, 26, 23])


def test_get_coord_3d(tmpdir, io_size):
    filename = os.path.join(tmpdir.strpath, "example.e")

    with exodus(
        filename,
        mode="w",
        title="Example",
        array_type="numpy",
        numDims=3,
        numNodes=5,
        numElems=6,
        numBlocks=1,
        numNodeSets=0,
        numSideSets=1,
        io_size=io_size["io_size"],
    ) as e:
        e.put_coords(
            xCoords=np.arange(5, dtype=np.float64),
            yCoords=np.arange(5, dtype=np.float64) * 2,
            zCoords=np.arange(5, dtype=np.float64) * 3,
        )

    with exodus(filename, mode="r") as e:
        with pytest.raises(ValueError) as f:
            e.get_coord(0)
        assert f.value.args[0] == "Invalid index. Coordinate bounds: [1, 5]."
        with pytest.raises(ValueError) as f:
            e.get_coord(6)
        assert f.value.args[0] == "Invalid index. Coordinate bounds: [1, 5]."

        np.testing.assert_allclose(e.get_coord(1), [0.0, 0.0, 0.0])
        np.testing.assert_allclose(e.get_coord(2), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(e.get_coord(3), [2.0, 4.0, 6.0])

        # Now all at once.
        np.testing.assert_allclose(
            e.get_coord([1, 2, 3]),
            [[0.0, 1.0, 2.0], [0.0, 2.0, 4.0], [0.0, 3.0, 6.0]],
        )


def test_get_coord_2d(tmpdir, io_size):
    filename = os.path.join(tmpdir.strpath, "example.e")

    with exodus(
        filename,
        mode="w",
        title="Example",
        array_type="numpy",
        numDims=2,
        numNodes=5,
        numElems=6,
        numBlocks=1,
        numNodeSets=0,
        numSideSets=1,
        io_size=io_size["io_size"],
    ) as e:
        e.put_coords(
            xCoords=np.arange(5, dtype=np.float64),
            yCoords=np.arange(5, dtype=np.float64) * 2,
            zCoords=np.zeros(5),
        )

    with exodus(filename, mode="r") as e:
        with pytest.raises(ValueError) as f:
            e.get_coord(0)
        assert f.value.args[0] == "Invalid index. Coordinate bounds: [1, 5]."
        with pytest.raises(ValueError) as f:
            e.get_coord(6)
        assert f.value.args[0] == "Invalid index. Coordinate bounds: [1, 5]."

        np.testing.assert_allclose(e.get_coord(1), [0.0, 0.0, 0.0])
        np.testing.assert_allclose(e.get_coord(2), [1.0, 2.0, 0.0])
        np.testing.assert_allclose(e.get_coord(3), [2.0, 4.0, 0.0])


def test_get_coords_3d(tmpdir, io_size):
    filename = os.path.join(tmpdir.strpath, "example.e")

    with exodus(
        filename,
        mode="w",
        title="Example",
        array_type="numpy",
        numDims=3,
        numNodes=5,
        numElems=6,
        numBlocks=1,
        numNodeSets=0,
        numSideSets=1,
        io_size=io_size["io_size"],
    ) as e:
        e.put_coords(
            xCoords=np.arange(5, dtype=np.float64),
            yCoords=np.arange(5, dtype=np.float64) * 2,
            zCoords=np.arange(5, dtype=np.float64) * 3,
        )

    with exodus(filename, mode="r") as e:
        np.testing.assert_allclose(
            e.get_coords(), [np.arange(5), np.arange(5) * 2, np.arange(5) * 3]
        )


def test_get_coords_2d(tmpdir, io_size):
    filename = os.path.join(tmpdir.strpath, "example.e")

    with exodus(
        filename,
        mode="w",
        title="Example",
        array_type="numpy",
        numDims=2,
        numNodes=5,
        numElems=6,
        numBlocks=1,
        numNodeSets=0,
        numSideSets=1,
        io_size=io_size["io_size"],
    ) as e:
        e.put_coords(
            xCoords=np.arange(5, dtype=np.float64),
            yCoords=np.arange(5, dtype=np.float64) * 2,
            zCoords=np.zeros(5),
        )

    with exodus(filename, mode="r") as e:
        np.testing.assert_allclose(
            e.get_coords(), [np.arange(5), np.arange(5) * 2, np.zeros(5)]
        )


def test_get_elem_connectivity_quad(tmpdir, io_size):
    filename = os.path.join(tmpdir.strpath, "example.e")

    # Generate test file.
    with exodus(
        filename,
        mode="w",
        title="Example",
        array_type="numpy",
        numDims=2,
        numNodes=5,
        numElems=6,
        numBlocks=1,
        numNodeSets=0,
        numSideSets=1,
        io_size=io_size["io_size"],
    ) as e:
        e.put_coords(
            xCoords=np.arange(5, dtype=np.float64),
            yCoords=np.arange(5, dtype=np.float64) * 2,
            zCoords=np.zeros(5),
        )
        e.put_elem_blk_info(1, "QUAD", 6, 4, 0)
        e.put_elem_connectivity(1, np.arange(6 * 4) + 7)

    with exodus(filename, mode="r") as e:
        conn, num_elem, num_nodes_per_elem = e.get_elem_connectivity(id=1)

    np.testing.assert_equal(conn, (np.arange(6 * 4) + 7).reshape((6, 4)))
    assert num_elem == 6
    assert num_nodes_per_elem == 4


def test_get_elem_connectivity_hex(tmpdir, io_size):
    filename = os.path.join(tmpdir.strpath, "example.e")

    # Generate test file.
    with exodus(
        filename,
        mode="w",
        title="Example",
        array_type="numpy",
        numDims=2,
        numNodes=5,
        numElems=6,
        numBlocks=1,
        numNodeSets=0,
        numSideSets=1,
        io_size=io_size["io_size"],
    ) as e:
        e.put_coords(
            xCoords=np.arange(5, dtype=np.float64),
            yCoords=np.arange(5, dtype=np.float64) * 2,
            zCoords=np.zeros(5),
        )
        e.put_elem_blk_info(1, "HEX", 3, 8, 0)
        e.put_elem_connectivity(1, np.arange(3 * 8) + 7)

    with exodus(filename, mode="r") as e:
        conn, num_elem, num_nodes_per_elem = e.get_elem_connectivity(id=1)

    np.testing.assert_equal(conn, (np.arange(3 * 8) + 7).reshape((3, 8)))
    assert num_elem == 3
    assert num_nodes_per_elem == 8


def test_get_elem_connectivity_only_some_indices(tmpdir, io_size):
    filename = os.path.join(tmpdir.strpath, "example.e")

    # Generate test file.
    with exodus(
        filename,
        mode="w",
        title="Example",
        array_type="numpy",
        numDims=2,
        numNodes=5,
        numElems=6,
        numBlocks=1,
        numNodeSets=0,
        numSideSets=1,
        io_size=io_size["io_size"],
    ) as e:
        e.put_coords(
            xCoords=np.arange(5, dtype=np.float64),
            yCoords=np.arange(5, dtype=np.float64) * 2,
            zCoords=np.zeros(5),
        )
        e.put_elem_blk_info(1, "HEX", 3, 8, 0)
        e.put_elem_connectivity(1, np.arange(3 * 8) + 7)

    with exodus(filename, mode="r") as e:
        conn, num_elem, num_nodes_per_elem = e.get_elem_connectivity(
            id=1, indices=[1, 3]
        )

    np.testing.assert_equal(
        conn, (np.arange(3 * 8) + 7).reshape((3, 8))[[0, 2]]
    )
    assert num_elem == 3
    assert num_nodes_per_elem == 8


def test_num_dims_accessor(tmpdir, io_size):
    # 2D
    filename = os.path.join(tmpdir.strpath, "example_2d.e")
    with exodus(
        filename,
        mode="w",
        title="Example",
        array_type="numpy",
        numDims=2,
        numNodes=5,
        numElems=6,
        numBlocks=1,
        numNodeSets=0,
        numSideSets=1,
        io_size=io_size["io_size"],
    ) as e:
        e.put_coords(
            xCoords=np.arange(5, dtype=np.float64),
            yCoords=np.arange(5, dtype=np.float64) * 2,
            zCoords=np.zeros(5),
        )
    with exodus(filename, mode="r") as e:
        assert e.num_dims == 2

    # 3D
    filename = os.path.join(tmpdir.strpath, "example_3d.e")
    with exodus(
        filename,
        mode="w",
        title="Example",
        array_type="numpy",
        numDims=3,
        numNodes=5,
        numElems=6,
        numBlocks=1,
        numNodeSets=0,
        numSideSets=1,
        io_size=io_size["io_size"],
    ) as e:
        e.put_coords(
            xCoords=np.arange(5, dtype=np.float64),
            yCoords=np.arange(5, dtype=np.float64) * 2,
            zCoords=np.zeros(5),
        )
    with exodus(filename, mode="r") as e:
        assert e.num_dims == 3


def test_get_element_variable_values(tmpdir, io_size):
    filename = os.path.join(tmpdir.strpath, "example.e")

    e = exodus(
        filename,
        mode="w",
        title="Example",
        array_type="numpy",
        numDims=3,
        numNodes=5,
        numElems=6,
        numBlocks=1,
        numNodeSets=0,
        numSideSets=1,
        io_size=io_size["io_size"],
    )
    e.set_element_variable_number(5)
    e.put_element_variable_name("random", 3)
    # requires an actual element block.
    e.put_elem_blk_info(1, "HEX", 6, 3, 0)
    e.put_element_variable_values(1, "random", 1, np.arange(6))
    e.close()

    with exodus(filename, mode="a") as e:
        values = e.get_element_variable_values(1, "random", 1)
        np.testing.assert_equal(values, np.arange(6))

        # Raises a value error if the variable does not exist.
        with pytest.raises(ValueError):
            e.get_element_variable_values(1, "rando", 1)


def test_get_node_variable_values(tmpdir, io_size):
    filename = os.path.join(tmpdir.strpath, "example.e")

    e = exodus(
        filename,
        mode="w",
        title="Example",
        array_type="numpy",
        numDims=3,
        numNodes=5,
        numElems=6,
        numBlocks=1,
        numNodeSets=0,
        numSideSets=1,
        io_size=io_size["io_size"],
    )
    e.set_node_variable_number(2)
    e.put_node_variable_name("good friend", 1)
    e.put_node_variable_values("good friend", 1, np.arange(5))

    np.testing.assert_equal(
        actual=e.get_node_variable_values(name="good friend", step=1),
        desired=np.arange(5),
    )

    # Invalid step.
    with pytest.raises(ValueError) as err:
        e.get_node_variable_values(name="good friend", step=0)
    assert err.value.args[0] == "Step must be larger than zero."

    with pytest.raises(ValueError) as err:
        e.get_node_variable_values(name="good friend", step=10)
    assert err.value.args[0] == "Step must be 0 < step <= 1."

    # Invalid name.
    with pytest.raises(ValueError) as err:
        e.get_node_variable_values(name="random", step=1)
    assert err.value.args[0] == "'random' is not in list"


def test_get_node_variable_names(tmpdir, io_size):
    filename = os.path.join(tmpdir.strpath, "example.e")

    e = exodus(
        filename,
        mode="w",
        title="Example",
        array_type="numpy",
        numDims=3,
        numNodes=5,
        numElems=6,
        numBlocks=1,
        numNodeSets=0,
        numSideSets=1,
        io_size=io_size["io_size"],
    )
    e.set_node_variable_number(2)
    e.put_node_variable_name("good friend", 1)

    assert e.get_node_variable_names() == ["good friend", ""]
