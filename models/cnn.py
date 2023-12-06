import torch
from torch import nn
import torch.nn.functional as F
import math

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from metadata.char import MAX_LENGTH as max_len
from metadata.char import INPUT_DIM as input_dim

CONV_DIM = 32
FC_DIM = 512
FC_DROPOUT = 0.2
WINDOW_WIDTH = 16
WINDOW_STRIDE = 8


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size = 3,
        stride = 1,
        padding = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        If default kernel,stride and padding are not changed
        channel: 
        input channel --> output channel
        H,W:
        output h,w remains same as the input 
        """
        c = self.conv(x)
        r = self.relu(c)
        return r


class CNN(nn.Module):
    def __init__(
        self,
        num_classes,
        conv_dim = None,   #conv dimension
        fc_dim = None,     #fully connected layer dimension
        fc_dropout = None, #fully connected layer dropout
        ww = None,         #window width
        ws = None,         #window stride
        limit_output = False #to limit output length
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.output_length = max_len

        C, H, _ = input_dim
        conv_dim = CONV_DIM if conv_dim is None else conv_dim
        fc_dim = FC_DIM if fc_dim is None else fc_dim
        fc_dropout = FC_DROPOUT if fc_dropout is None else fc_dropout
        self.WW = WINDOW_WIDTH if ww is None else ww
        self.WS = WINDOW_STRIDE if ws is None else ws
        self.limit_output_length = limit_output

        self.convs = nn.Sequential(
            ConvBlock(C, conv_dim),
            ConvBlock(conv_dim, conv_dim),
            ConvBlock(conv_dim, conv_dim, stride=2),
            ConvBlock(conv_dim, conv_dim),
            ConvBlock(conv_dim, conv_dim * 2, stride=2),
            ConvBlock(conv_dim * 2, conv_dim * 2),
            ConvBlock(conv_dim * 2, conv_dim * 4, stride=2),
            ConvBlock(conv_dim * 4, conv_dim * 4),
            ConvBlock(
                conv_dim * 4, fc_dim, kernel_size=(H // 8, self.WW // 8), stride=(H // 8, self.WS // 8), padding=0
            ),
        )
        self.fc1 = nn.Linear(fc_dim, fc_dim)
        self.dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(fc_dim, self.num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Conv2d,
                nn.Conv3d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
                nn.Linear,
            }:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    _fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x
            (B, C, H, W) input image

        Returns
        -------
        torch.Tensor
            (B, Cl, S) logits, where S is the length of the sequence and Cl is the number of classes
            S can be computed from W and self.window_width
            Cl is self.num_classes
        """
        x = self.convs(x) 
        x = x.squeeze(2).permute(0, 2, 1) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # (B, S, C)
        x = x.permute(0, 2, 1)  # (B, C, S)
        if self.limit_output_length:
            x = x[:, :, : self.output_length]
        return x
