# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Reference: https://github.com/tenstorrent/tt-metal/tree/main/models/demos/yolov4/reference

from .downsample1 import DownSample1
from .downsample2 import DownSample2
from .downsample3 import DownSample3
from .downsample4 import DownSample4
from .downsample5 import DownSample5
from .neck import Neck
from .head import Head

import torch
import torch.nn as nn


class Yolov4(nn.Module):
    def __init__(self):
        super(Yolov4, self).__init__()
        self.downsample1 = DownSample1()
        self.downsample2 = DownSample2()
        self.downsample3 = DownSample3()
        self.downsample4 = DownSample4()
        self.downsample5 = DownSample5()
        self.neck = Neck()
        self.head = Head()

    def forward(self, input: torch.Tensor):
        d1 = self.downsample1(input)
        d2 = self.downsample2(d1)
        d3 = self.downsample3(d2)
        d4 = self.downsample4(d3)
        d5 = self.downsample5(d4)
        x20, x13, x6 = self.neck(d5, d4, d3)
        x4, x5, x6 = self.head(x20, x13, x6)

        return x4, x5, x6

    @staticmethod
    def from_random_weights():
        model = Yolov4()
        model.eval()

        new_state_dict = {}
        for name, parameter in model.state_dict().items():
            if isinstance(parameter, torch.FloatTensor):
                new_state_dict[name] = parameter

        model.load_state_dict(new_state_dict)
        return model
