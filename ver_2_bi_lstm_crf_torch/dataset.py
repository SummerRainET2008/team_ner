from ver_2_bi_lstm_crf_torch import *
from ver_2_bi_lstm_crf_torch.param import Param
import torch.utils.data

class _Dataset(torch.utils.data.Dataset):
  def __init__(self, data_files: list):
    data = []
    for f in data_files:
      data.extend(pickle.load(open(f, "rb")))
    self._data = data

  def __len__(self):
    return len(self._data)

  def __getitem__(self, index):
    return self._data[index]

def _pad_batch_data(batch):
  batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
  word_ids, labels, mask_size = list(zip(*batch))
  word_ids = torch.LongTensor(word_ids)
  labels = torch.LongTensor(labels)

  return word_ids, labels

def get_batch_data(data_files, epoch_num, shuffle: bool):
  param = Param()
  dataset = _Dataset(data_files)
  data_iter = torch.utils.data.DataLoader(
    dataset, param._batch_size, shuffle=shuffle, num_workers=0,
    collate_fn=lambda x: x,
  )

  for epoch_id in range(epoch_num):
    for batch in data_iter:
      yield epoch_id, _pad_batch_data(batch)
