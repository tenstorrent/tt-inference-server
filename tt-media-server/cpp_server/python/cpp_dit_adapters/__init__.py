# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The only Python of the C++ video worker: the calls into tt_dit.

The C++ server owns everything else (runner classes, mesh and fabric, request
validation, the arguments of each call, warmup, mp4 export).
It reaches the pipelines through this package:

    adapter = create("wan_t2v", mesh_device, options)   # builds the pipeline
    result = adapter.call(args)                         # one generation

`options` and `args` are plain dicts built by the C++ model of the same name
(src/runtime/dit/models_wan.cpp), which says what each key means. A result is a
dict:

    {"kind": "frames", "frames": ndarray}            video frames
    {"kind": "av", "frames", "audio", "sampling_rate", "fps", "pixel_format"}
    {"kind": "path", "path": str}                    a file the pipeline wrote
    {"kind": "images", "images": [png bytes]}        images

Arrays are C-contiguous uint8, float32 or float64. Each module imports tt_dit
inside create(), so importing this package needs neither ttnn nor tt-metal.

Add a model's adapter with @register("name") on a function
create(mesh_device, options) -> Adapter.
"""

from __future__ import annotations

from typing import Any, Callable

from ._common import Adapter

_FACTORIES: dict[str, Callable[[Any, dict], Adapter]] = {}


def register(name: str):
    def decorate(factory):
        if name in _FACTORIES:
            raise ValueError(f"adapter {name!r} is already registered")
        _FACTORIES[name] = factory
        return factory

    return decorate


def adapter_names() -> list[str]:
    return sorted(_FACTORIES)


def create(name: str, mesh_device: Any, options: dict) -> Adapter:
    try:
        factory = _FACTORIES[name]
    except KeyError:
        raise ValueError(
            f"no adapter {name!r}; known: {', '.join(adapter_names())}"
        ) from None
    return factory(mesh_device, options)


from . import wan  # noqa: E402,F401  (register)
