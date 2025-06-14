import torch
from torch import nn
from torchvision import models
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        x = x + self.pe[:, :x.size(1)]
        return x

class HandwritingTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        vocab_size: int,
        d_model: int = 128,
        nhead_en: int = 1,
        num_layers_en: int = 1,
        nhead_de: int = 1,
        num_layers_de: int = 1,
        dropout: float = 0.2  # Added dropout parameter
    ):
        super(HandwritingTransformer, self).__init__()

        self.cnn = models.mobilenet_v2(pretrained=True).features[:4]
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.input_size = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, 200)

        # Add LayerNorm and Dropout after input_size
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead_en, dropout=dropout, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers_en)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead_de, dropout=dropout, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers_de)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_decoder = PositionalEncoding(d_model, 20)

        # Add LayerNorm and Dropout after embedding
        self.embedding_norm = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(0.5)

        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        src = torch.stack([src[:, 0]] * 3, dim=1)
        src = self.cnn(src)

        src = src.flatten(1, 2)
        src = src.permute(0, 2, 1)
        src = self.input_size(src)

        # Apply LayerNorm and Dropout after input_size
        src = self.input_norm(src)
        src = self.input_dropout(src)

        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)
        memory = self.encoder(src)

        tgt = self.embedding(tgt)

        # Apply LayerNorm and Dropout after embedding
        tgt = self.embedding_norm(tgt)
        tgt = self.embedding_dropout(tgt)

        tgt = self.pos_decoder(tgt)
        tgt = tgt.permute(1, 0, 2)

        tgt_mask = self.generate_square_subsequent_mask(
            tgt.size(0)).to(tgt.device)

        output: torch.Tensor = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        output = self.output(output)
        output = output.permute(1, 0, 2)
        return output

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)