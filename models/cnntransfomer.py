import math

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

import torch
from torch import nn

from models.cnn import CNN
from models.transformer_util import generate_square_subsequent_mask, PositionalEncoding

from metadata.char import MAPPING as mapping
from metadata.char import MAX_LENGTH as max_len
from metadata.char import INPUT_DIM as input_dim

TF_DIM = 256
TF_FC_DIM = 256
TF_DROPOUT = 0.4
TF_LAYERS = 4
TF_NHEAD = 4


class CNNTransformer(nn.Module):
    def __init__(
        self,
        tf_dim = None,
        tf_fc_dim = None,
        tf_nh = None,
        tf_dropout = None,
        tf_layers = None
    ):
        super().__init__()
        self.input_dims = input_dim
        self.num_classes = len(mapping)
        inverse_mapping = {val: ind for ind, val in enumerate(mapping)}
        self.start_token = inverse_mapping["<S>"]
        self.end_token = inverse_mapping["<E>"]
        self.padding_token = inverse_mapping["<P>"]
        self.max_output_length = max_len


        self.dim = TF_DIM if tf_dim is None else tf_dim
        tf_fc_dim = TF_FC_DIM if tf_fc_dim is None else tf_fc_dim
        tf_nhead = TF_NHEAD if tf_nh is None else tf_nh
        tf_dropout = TF_DROPOUT if tf_dropout is None else tf_dropout
        tf_layers = TF_LAYERS if tf_layers is None else tf_layers

 
        self.line_cnn = CNN(num_classes=self.dim)
        #CNN outputs (B, E, S) log probs, with E == dim

        self.embedding = nn.Embedding(self.num_classes, self.dim)
        self.fc = nn.Linear(self.dim, self.num_classes)

        self.pos_encoder = PositionalEncoding(d_model=self.dim)

        self.y_mask = generate_square_subsequent_mask(self.max_output_length)

        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.dim, nhead=tf_nhead, dim_feedforward=tf_fc_dim, dropout=tf_dropout),
            num_layers=tf_layers,
        )

        self.init_weights()  # This is empirically important

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode each image tensor in a batch into a sequence of embeddings.

        Parameters
        ----------
        x
            (B, H, W) image

        Returns
        -------
        torch.Tensor
            (Sx, B, E) logits
        """
        x = self.line_cnn(x)  # (B, E, Sx)
        x = x * math.sqrt(self.dim)
        x = x.permute(2, 0, 1)  # (Sx, B, E)
        x = self.pos_encoder(x)  # (Sx, B, E)
        return x

    def decode(self, x, y):
        """Decode a batch of encoded images x using preceding ground truth y.

        Parameters
        ----------
        x
            (Sx, B, E) image encoded as a sequence
        y
            (B, Sy) with elements in [0, C-1] where C is num_classes

        Returns
        -------
        torch.Tensor
            (Sy, B, C) logits
        """
        y_padding_mask = y == self.padding_token
        y = y.permute(1, 0)  # (Sy, B)
        y = self.embedding(y) * math.sqrt(self.dim)  # (Sy, B, E)
        y = self.pos_encoder(y)  # (Sy, B, E)
        Sy = y.shape[0]
        y_mask = self.y_mask[:Sy, :Sy].type_as(x)
        output = self.transformer_decoder(
            tgt=y, memory=x, tgt_mask=y_mask, tgt_key_padding_mask=y_padding_mask
        )  # (Sy, B, E)
        output = self.fc(output)  # (Sy, B, C)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict sequences of tokens from input images auto-regressively.

        Parameters
        ----------
        x
            (B, H, W) image

        Returns
        -------
        torch.Tensor
            (B, Sy) with elements in [0, C-1] where C is num_classes
        """
        B = x.shape[0]
        S = self.max_output_length
        x = self.encode(x)  # (Sx, B, E)

        output_tokens = (torch.ones((B, S)) * self.padding_token).type_as(x).long()  # (B, S)
        output_tokens[:, 0] = self.start_token  # Set start token
        for Sy in range(1, S):
            y = output_tokens[:, :Sy]  # (B, Sy)
            output = self.decode(x, y)  # (Sy, B, C)
            output = torch.argmax(output, dim=-1)  # (Sy, B)
            output_tokens[:, Sy] = output[-1:]  # Set the last output token

        # Set all tokens after end token to be padding
        for Sy in range(1, S):
            ind = (output_tokens[:, Sy - 1] == self.end_token) | (output_tokens[:, Sy - 1] == self.padding_token)
            output_tokens[ind, Sy] = self.padding_token

        return output_tokens  # (B, Sy)