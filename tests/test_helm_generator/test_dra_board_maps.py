# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from workflows.device_utils import dra_device_board_counts, dra_device_board_names


def test_board_count_and_name_maps_cover_identical_devices():
    """The ResourceClaim template reads count from deviceBoardCounts and boardName
    from deviceBoardNames; if a device had one but not the other, it would render an
    unsatisfiable `boardName == ""` selector. Both maps derive from the same
    generator, so their key sets must always match.
    """
    assert set(dra_device_board_counts()) == set(dra_device_board_names())


def test_board_names_translate_galaxy_and_pass_through_others():
    names = dra_device_board_names()
    # Galaxy UBB is named differently by the driver than the tt-smi board_type.
    assert names["galaxy"] == "galaxy-wormhole"
    # Multi-board device shapes select by their single board type.
    assert names["t3k"] == "n300"
    assert names["n300"] == "n300"
