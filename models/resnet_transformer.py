import math
import pretrainedmodels

import torch
from torch import nn
import torchvision

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from models.transformer_util import generate_square_subsequent_mask, PositionalEncoding, PositionalEncodingImage

from metadata.char import MAX_LENGTH as max_len
from metadata.char import INPUT_DIM as input_dim
from metadata.char import MAPPING as mapping

TF_DIM = 256
TF_FC_DIM = 1024
TF_DROPOUT = 0.4
TF_LAYERS = 4
TF_NHEAD = 4
RESNET_DIM = 512  # hard-coded


class ResnetTransformer(nn.Module):
    """Pass an image through a Resnet and decode the resulting embedding with a Transformer."""

    def __init__(
        self,
        tf_dim = None,
        tf_fc_dim = None,
        tf_nh = None,
        tf_dropout = None,
        tf_layers = None
    ) -> None:
        super().__init__()

        self.input_dims = input_dim
        self.num_classes = len(mapping)

        self.mapping = mapping
        inverse_mapping = {val: ind for ind, val in enumerate(self.mapping)}
        self.start_token = inverse_mapping["<S>"]
        self.end_token = inverse_mapping["<E>"]
        self.padding_token = inverse_mapping["<P>"]
        self.max_output_length = max_len


        self.dim = TF_DIM if tf_dim is None else tf_dim
        tf_fc_dim = TF_FC_DIM if tf_fc_dim is None else tf_fc_dim
        tf_nhead = TF_NHEAD if tf_nh is None else tf_nh
        tf_dropout = TF_DROPOUT if tf_dropout is None else tf_dropout
        tf_layers = TF_LAYERS if tf_layers is None else tf_layers

        
        self.resnet = pretrainedmodels.__dict__["resnet34"](pretrained = None)  
        # Resnet will output (B, RESNET_DIM, _H, _W) logits where _H = input_H // 32, _W = input_W // 32

        self.encoder_projection = nn.Conv2d(RESNET_DIM, self.dim, kernel_size=1)
        # encoder_projection will output (B, dim, _H, _W) logits

        self.enc_pos_encoder = PositionalEncodingImage(
            d_model=self.dim, max_h=self.input_dims[1], max_w=self.input_dims[2]
        )  # Max (Ho, Wo)

        # ## Decoder part
        self.embedding = nn.Embedding(self.num_classes, self.dim)
        self.fc = nn.Linear(self.dim, self.num_classes)

        self.dec_pos_encoder = PositionalEncoding(d_model=self.dim, max_len=self.max_output_length)

        self.y_mask = generate_square_subsequent_mask(self.max_output_length)

        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.dim, nhead=tf_nhead, dim_feedforward=tf_fc_dim, dropout=tf_dropout),
            num_layers=tf_layers,
        )

        self.init_weights()  # This is empirically important

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Autoregressively produce sequences of labels from input images.

        Parameters
        ----------
        x
            (B, Ch, H, W) image, where Ch == 1 or Ch == 3

        Returns
        -------
        output_tokens
            (B, Sy) with elements in [0, C-1] where C is num_classes
        """
        B = x.shape[0]
        S = self.max_output_length
        x = self.encode(x)  # (Sx, B, E)

        output_tokens = (torch.ones((B, S)) * self.padding_token).type_as(x).long()  # (B, Sy)
        output_tokens[:, 0] = self.start_token  # Set start token
        for Sy in range(1, S):
            y = output_tokens[:, :Sy]  # (B, Sy)
            output = self.decode(x, y)  # (Sy, B, C)
            output = torch.argmax(output, dim=-1)  # (Sy, B)
            output_tokens[:, Sy] = output[-1]  # Set the last output token

            # Early stopping of prediction loop to speed up prediction
            if ((output_tokens[:, Sy] == self.end_token) | (output_tokens[:, Sy] == self.padding_token)).all():
                break

        # Set all tokens after end or padding token to be padding
        for Sy in range(1, S):
            ind = (output_tokens[:, Sy - 1] == self.end_token) | (output_tokens[:, Sy - 1] == self.padding_token)
            output_tokens[ind, Sy] = self.padding_token

        return output_tokens  # (B, Sy)

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

        nn.init.kaiming_normal_(self.encoder_projection.weight.data, a=0, mode="fan_out", nonlinearity="relu")
        if self.encoder_projection.bias is not None:
            _fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.encoder_projection.weight.data)
            bound = 1 / math.sqrt(fan_out)
            nn.init.normal_(self.encoder_projection.bias, -bound, bound)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode each image tensor in a batch into a sequence of embeddings.

        Parameters
        ----------
        x
            (B, Ch, H, W) image, where Ch == 1 or Ch == 3

        Returns
        -------
            (Sx, B, E) sequence of embeddings, going left-to-right, top-to-bottom from final ResNet feature maps
        """
        _B, C, _H, _W = x.shape
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.resnet.features(x)  # (B, RESNET_DIM, _H // 32, _W // 32),   
        x = self.encoder_projection(x)  # (B, E, _H // 32, _W // 32),   

        x = self.enc_pos_encoder(x)  # (B, E, Ho, Wo);     Ho = _H // 32, Wo = _W // 32
        x = torch.flatten(x, start_dim=2)  # (B, E, Ho * Wo)
        x = x.permute(2, 0, 1)  # (Sx, B, E);    Sx = Ho * Wo
        return x

    def decode(self, x, y):
        """Decode a batch of encoded images x with guiding sequences y.

        During autoregressive inference, the guiding sequence will be previous predictions.

        During training, the guiding sequence will be the ground truth.

        Parameters
        ----------
        x
            (Sx, B, E) images encoded as sequences of embeddings
        y
            (B, Sy) guiding sequences with elements in [0, C-1] where C is num_classes

        Returns
        -------
        torch.Tensor
            (Sy, B, C) batch of logit sequences
        """
        y_padding_mask = y == self.padding_token
        y = y.permute(1, 0)  # (Sy, B)
        y = self.embedding(y) * math.sqrt(self.dim)  # (Sy, B, E)
        y = self.dec_pos_encoder(y)  # (Sy, B, E)
        Sy = y.shape[0]
        y_mask = self.y_mask[:Sy, :Sy].type_as(x)
        output = self.transformer_decoder(
            tgt=y, memory=x, tgt_mask=y_mask, tgt_key_padding_mask=y_padding_mask
        )  # (Sy, B, E)
        output = self.fc(output)  # (Sy, B, C)
        return output