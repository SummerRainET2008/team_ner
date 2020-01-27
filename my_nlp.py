import torch
from torch import nn


class BiLSTM(nn.Module):
  def __init__(self, hidden_size, dropout=0.5, num_layers=2):
    super(BiLSTM, self).__init__()
    self._enc_layer = nn.LSTM(
      input_size=hidden_size, hidden_size=hidden_size // 2,
      num_layers=num_layers, dropout=dropout, bidirectional=True
    )

  def forward(self, input):
    output, _ = self._enc_layer(input)
    return output

