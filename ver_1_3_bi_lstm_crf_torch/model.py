import torch
from torch import nn
from torchcrf import CRF
from ver_1_3_bi_lstm_crf_torch import *
from pa_nlp.pytorch import nlp_torch

class BiLSTM_CRF(nn.Module):
  def __init__(self, max_seq_len, tag_size, vob_size, embedding_size, LSTM_layer_num, dropout):
    super(BiLSTM_CRF, self).__init__()
    self._word_embeds = nn.Embedding(vob_size, embedding_size)
    self._hidden_size = embedding_size
    self._hidden = self._init_hidden()

    self._bi_LSTM = BiLSTM(
      hidden_size=self._hidden_size,
      dropout=dropout,
      num_layers=LSTM_layer_num
    )
    self._dense = nlp_torch.Dense(
      nn.Linear(self._hidden_size, tag_size),
      None
    )
    self._CRF = CRF(tag_size)
    self._init_weights()

  def _init_hidden(self):
    return (torch.randn(2, 1, self._hidden_size),
            torch.randn(2, 1, self._hidden_size))

  def _init_weights(self):
    for name, w in self.named_parameters():
      print(name)
      if "CRF" in name:
        continue
      if "dense" in name:
        if "weight" in name:
          nn.init.xavier_normal_(w)
        elif "bias" in name:
          nn.init.zeros_(w)
      if "bi_LSTM" in name:
        if "weight" in name:
          nn.init.xavier_normal_(w)
        elif "bias" in name:
          nn.init.zeros_(w)


  def neg_log_likelihood(self, input_x, input_y):
    embeds = self._word_embeds(input_x)
    embeds = embeds.view(embeds.size()[1], embeds.size()[0], embeds.size()[2])
    lstm_out = self._bi_LSTM(embeds)
    states_to_tag = self._dense(lstm_out)
    tags = input_y.view(input_y.size()[1], input_y.size()[0])
    log_likelihood = self._CRF(states_to_tag, tags)
    loss = -log_likelihood.mean()
    return loss

  def forward(self, input_x):
    embeds = self._word_embeds(input_x)
    embeds = embeds.view(embeds.size()[1], embeds.size()[0], embeds.size()[2])
    lstm_out= self._bi_LSTM(embeds)
    states_to_tag = self._dense(lstm_out)
    opt_seq = self._CRF.decode(states_to_tag)
    return opt_seq


