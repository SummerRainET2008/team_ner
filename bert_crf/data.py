#coding: utf8
#author: Tian Xia (SummerRainET2008@gmail.com)

import os
from pa_nlp.common import *
from pa_nlp.tf_1x.bert.bert_tokenizer import *
'''The format of data is defined as
Each line is a python dict string, denoting a sample dict.
{
text: '...',
tags = [[pos_from, pos_to, tag_name, source_text] ... ]
}
'''

def create_parameter(
  train_file,
  vali_file,  # can be None
  vob_file,
  tag_list,   # ["person", "organization", ...]
  max_seq_length=64,
  epoch_num=1,
  batch_size=1024,
  embedding_size=128,
  RNN_type="lstm",
  LSTM_layer_num=1,
  dropout_keep_prob=0.5,
  learning_rate=0.001,
  l2_reg_lambda=0.0,
  evaluate_frequency=100,  # must divided by 100.
  remove_OOV=True,
  GPU: int=-1,  # which_GPU_to_run: [0, 4), and -1 denote CPU.
  model_dir: str= "model",
  bert_config_file='',
  bert_init_ckpt='',
  ):
  
  assert os.path.isfile(train_file)
  assert os.path.isfile(vali_file)
  
  tag_list = ["O"] + sorted(set(tag_list) - set("O"))
  print(f"tag_list: {tag_list}")
  
  return {
    "train_file": os.path.realpath(train_file),
    "vali_file": os.path.realpath(vali_file),
    "vob_file": os.path.realpath(vob_file),
    "tag_list": tag_list,
    "max_seq_length": max_seq_length,
    "epoch_num": epoch_num,
    "batch_size": batch_size,
    "embedding_size": embedding_size,
    "RNN_type": RNN_type,
    "LSTM_layer_num": LSTM_layer_num,
    "learning_rate": learning_rate,
    "dropout_keep_prob": dropout_keep_prob,
    "l2_reg_lambda": l2_reg_lambda,
    "evaluate_frequency":  evaluate_frequency,
    "remove_OOV": remove_OOV,
    "GPU":  GPU,
    "model_dir": os.path.realpath(model_dir),
    "bert_config_file": bert_config_file,
    "bert_init_ckpt": bert_init_ckpt,
  }

class DataSet:
  def __init__(self, data_file, tag_list, max_seq_len, vob_file):
    self.vob_file = vob_file
    self._tag_list = tag_list
    samples = read_pydict_file(data_file)
    self.max_seq_len = max_seq_len
    self._data = [self._gen_label(sample) for sample in samples]
    self._data_name = os.path.basename(data_file)


  def _gen_label(self, sample):
    word_list = sample["word_list"]
    original_text = sample["text"]
    T = BertTokenizer(vocab_file=self.vob_file)
    tokens, input_ids, input_mask, segment_ids = T.parse_single_sentence(
      text_a=original_text, max_seq_length=self.max_seq_len
    )
    labels = [0] * self.max_seq_len
    for key, value in sample["tags"].items():
      for pos in range(value[0], value[1]+1):
        if pos >= self.max_seq_len:
          break
        if pos == value[0]:
          labels[pos] = self._tag_list.index(key) * 2 - 1
        else:
          labels[pos] = labels[value[0]] + 1

    return input_ids[:self.max_seq_len], input_mask[:self.max_seq_len], segment_ids[:self.max_seq_len], labels, min(len(word_list), self.max_seq_len)


  def size(self):
    return len(self._data)
      
  def create_batch_iter(self, batch_size, epoch_num, shuffle: bool):
    return create_batch_iter_helper1(self._data_name, self._data,
                                    batch_size, epoch_num, shuffle)

