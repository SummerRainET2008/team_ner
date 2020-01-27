import torch
from torch import nn
from torchcrf import CRF
from bi_lstm_crf_torch import *
from pa_nlp.pytorch import nlp_torch

class BiLSTM_CRF(nn.Module):
  def __init__(self, max_seq_len, tag_size, vob_size, embedding_size, LSTM_layer_num, dropout):
    super(BiLSTM_CRF, self).__init__()
    self.word_embeds = nn.Embedding(vob_size, embedding_size)
    self.hidden_size = embedding_size
    self.hidden = self._init_hidden()

    self._bi_LSTM = BiLSTM(
      hidden_size=self.hidden_size,
      dropout=dropout,
      num_layers=LSTM_layer_num
    )
    self._dense = nlp_torch.Dense(
      nn.Linear(self.hidden_size, tag_size),
      None
    )
    self._CRF = CRF(tag_size)
    self._init_weights()

  def _init_hidden(self):
    return (torch.randn(2, 1, self.hidden_size),
            torch.randn(2, 1, self.hidden_size))

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
    print(input_x.size())
    embeds = self.word_embeds(input_x)
    embeds = embeds.view(embeds.size()[1], embeds.size()[0], embeds.size()[2])
    print(embeds.size())
    lstm_out = self._bi_LSTM(embeds)
    print(lstm_out.size())
    states_to_tag = self._dense(lstm_out)
    tags = input_y.view(input_y.size()[1], input_y.size()[0])
    log_likelihood = self._CRF(states_to_tag, tags)
    loss = -log_likelihood.mean()
    return loss

  def forward(self, input_x):
    print(input_x.size())
    embeds = self.word_embeds(input_x)
    embeds = embeds.view(embeds.size()[1], embeds.size()[0], embeds.size()[2])
    print(embeds.size())
    lstm_out= self._bi_LSTM(embeds)
    print(lstm_out.size())
    states_to_tag = self._dense(lstm_out)
    print(states_to_tag.size())
    opt_seq = self._CRF.decode(states_to_tag)
    return opt_seq


if __name__ == '__main__':
  model = BiLSTM_CRF(10, 4, 10, 8, 2, 0.5)
  print(model)
  test_loss= model.neg_log_likelihood(torch.tensor([[1, 2, 3, 4], [2,3,4,5]]), torch.tensor([[0, 1, 2, 1], [0,0,1,2]]))
  test_seq = model(torch.tensor([[1, 2, 3, 4]]))
  print(test_loss)
  print(test_seq)
