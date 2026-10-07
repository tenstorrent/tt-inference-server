# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Decode DRAM-bandwidth roofline for the benchmark tables.

Decode on a Tenstorrent device is DRAM-bandwidth bound: every output token
streams the full set of weights resident on each chip out of that chip's
DRAM. The floor on time-per-output-token is therefore

    floor_ms = weight_bytes_per_chip / dram_bw_per_chip
    weight_bytes_per_chip = param_count * bytes_per_param / tp_chips

and a measured TPOT is reported as ``pct_of_dram_floor = floor_ms / tpot * 100``
(100 % = at the floor, lower is worse). The column is informational only: it
never feeds a target check, and when the floor cannot be computed (unknown
device, no parameter count) the value is ``None``, which renders as "N/A".

Hardware constants live here, keyed by ``DeviceTypes`` *name* so the module
stays import-free (``workflows.workflow_types`` transitively imports
``workflow_module``, which imports ``report_module``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Per-chip DRAM bandwidth in GB/s. Wormhole (N150/N300/T3K/Galaxy) chips carry
# 12 GB of GDDR6 at 288 GB/s; Blackhole (P150/P300) chips carry 32 GB of GDDR6
# at 512 GB/s.
DRAM_BW_GBPS: Dict[str, float] = {
    "N150": 288.0,
    "N300": 288.0,
    "N150X4": 288.0,
    "T3K": 288.0,
    "GALAXY": 288.0,
    "GALAXY_T3K": 288.0,
    "DUAL_GALAXY": 288.0,
    "QUAD_GALAXY": 288.0,
    "P100": 512.0,
    "P150": 512.0,
    "P150X4": 512.0,
    "P150X8": 512.0,
    "P300": 512.0,
    "P300X2": 512.0,
    "BLACKHOLE_GALAXY": 512.0,
}

# Chips a single (DP=1) model instance is tensor-parallel across, i.e. how many
# ways the weights are split. ``DeviceTypes`` carries no chip counts, so they
# are spelled out here. A data-parallel deployment divides this by the number
# of engines (see :func:`tp_chips`).
TP_CHIPS: Dict[str, int] = {
    "N150": 1,
    "N300": 2,
    "N150X4": 4,
    "T3K": 8,
    "GALAXY": 32,
    "GALAXY_T3K": 8,
    "DUAL_GALAXY": 64,
    "QUAD_GALAXY": 128,
    "P100": 1,
    "P150": 1,
    "P150X4": 4,
    "P150X8": 8,
    "P300": 2,
    "P300X2": 4,
    "BLACKHOLE_GALAXY": 32,
}

# Block-float-8 (bfp8): 8-bit mantissa plus a shared 8-bit exponent per 16
# values -> 1 + 1/16 bytes per parameter. The dominant TT weight format;
# override per device spec with ``DeviceModelSpec.weight_bytes_per_param``.
DEFAULT_WEIGHT_BYTES_PER_PARAM = 1.0625


def _device_name(device: Any) -> str:
    """Upper-case ``DeviceTypes`` name from an enum member or a string."""
    name = getattr(device, "name", None)
    if not isinstance(name, str):
        name = str(device or "")
    return name.strip().upper()


def dram_bw_gbps(device: Any) -> Optional[float]:
    """Per-chip DRAM bandwidth in GB/s, or ``None`` for an unknown device."""
    return DRAM_BW_GBPS.get(_device_name(device))


def tp_chips(device: Any, data_parallel_size: Optional[int] = None) -> Optional[int]:
    """Chips one engine shards its weights across, or ``None`` when unknown.

    With ``data_parallel_size`` > 1 each engine owns ``chips / dp`` chips. When
    ``device`` is a ``DeviceTypes`` member its own ``get_data_parallel_subdevice``
    mapping is preferred (e.g. BLACKHOLE_GALAXY at DP=4 -> P150X8 -> 8 chips);
    the plain division is the fallback for string devices and unmapped pairs.
    """
    chips = TP_CHIPS.get(_device_name(device))
    if chips is None:
        return None
    dp = _positive_int(data_parallel_size)
    if dp is None or dp == 1:
        return chips
    subdevice_of = getattr(device, "get_data_parallel_subdevice", None)
    if callable(subdevice_of):
        try:
            sub_chips = TP_CHIPS.get(_device_name(subdevice_of(dp)))
        except ValueError:
            sub_chips = None
        if sub_chips is not None:
            return sub_chips
    if chips % dp:
        return None
    return chips // dp


def dram_floor_ms(
    param_count_billions: Any,
    device: Any,
    *,
    data_parallel_size: Optional[int] = None,
    weight_bytes_per_param: Optional[float] = None,
) -> Optional[float]:
    """Decode TPOT floor in ms for ``param_count_billions`` params on ``device``.

    ``floor_ms = params * bytes_per_param / tp_chips / dram_bw`` with params in
    units of 1e9 and bandwidth in GB/s (1e9 B/s), so the 1e9 factors cancel
    and the result only needs the s -> ms scale. ``None`` when any input is
    missing or non-positive.
    """
    params = _positive_float(param_count_billions)
    bytes_per_param = (
        DEFAULT_WEIGHT_BYTES_PER_PARAM
        if weight_bytes_per_param is None
        else _positive_float(weight_bytes_per_param)
    )
    bandwidth = dram_bw_gbps(device)
    chips = tp_chips(device, data_parallel_size)
    if params is None or bytes_per_param is None or bandwidth is None or not chips:
        return None
    return params * bytes_per_param / chips / bandwidth * 1000.0


def pct_of_dram_floor(floor_ms: Any, mean_tpot_ms: Any) -> Optional[float]:
    """``floor_ms / mean_tpot_ms * 100``; ``None`` when either side is unusable."""
    floor = _positive_float(floor_ms)
    tpot = _positive_float(mean_tpot_ms)
    if floor is None or tpot is None:
        return None
    return floor / tpot * 100.0


def dram_floor_ms_for_spec(model_spec: Any, device: Any = None) -> Optional[float]:
    """Floor for a ``ModelSpec``: its ``param_count`` (billions, inferred from
    the HF repo name when unset), the device it runs on, the engine count from
    ``vllm_args.data_parallel_size`` and the optional catalog override
    ``DeviceModelSpec.weight_bytes_per_param``.
    """
    if model_spec is None:
        return None
    device_model_spec = getattr(model_spec, "device_model_spec", None)
    if device is None:
        device = getattr(model_spec, "device_type", None)
    vllm_args = getattr(device_model_spec, "vllm_args", None) or {}
    return dram_floor_ms(
        getattr(model_spec, "param_count", None),
        device,
        data_parallel_size=_positive_int(vllm_args.get("data_parallel_size")),
        weight_bytes_per_param=getattr(
            device_model_spec, "weight_bytes_per_param", None
        ),
    )


def _positive_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number <= 0:
        return None
    return number


def _positive_int(value: Any) -> Optional[int]:
    number = _positive_float(value)
    if number is None or number != int(number):
        return None
    return int(number)


__all__ = [
    "DEFAULT_WEIGHT_BYTES_PER_PARAM",
    "DRAM_BW_GBPS",
    "TP_CHIPS",
    "dram_bw_gbps",
    "dram_floor_ms",
    "dram_floor_ms_for_spec",
    "pct_of_dram_floor",
    "tp_chips",
]
