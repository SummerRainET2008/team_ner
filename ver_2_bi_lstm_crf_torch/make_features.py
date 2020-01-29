from ver_2_bi_lstm_crf_torch import *
import pickle, os
from ver_2_bi_lstm_crf_torch.param import Param
from pytorch_transformers import RobertaTokenizer

class Tokenizer:
  _inst = None

  def __init__(self):
    param = Param()
    inst = RobertaTokenizer.from_pretrained(
      param.pretrained_model
    )
    self._tokenizer = inst
    self._cls_idx = inst.convert_tokens_to_ids(inst.cls_token)
    self._sep_idx = inst.convert_tokens_to_ids(inst.sep_token)
    self._pad_idx = inst.convert_tokens_to_ids(inst.pad_token)
    print(f"cls_id: {self._cls_idx}")
    print(f"sep_id: {self._sep_idx}")
    print(f"pad_id: {self._pad_idx}")

  @staticmethod
  def get_instance():
    if Tokenizer._inst is None:
      Tokenizer._inst = Tokenizer()
    return Tokenizer._inst

  def tokenize(self, sentence, max_len):
    ids = self._tokenizer.encode(sentence)
    word_ids = [self._cls_idx] + ids + [self._sep_idx]
    word_ids = word_ids[:max_len]

    return word_ids

  def get_vob_size(self):
    return len(self._tokenizer)

def process(data_files: list, out_file: str, param: Param):
  def data_generator():
    tokenizer = Tokenizer.get_instance()
    print(tokenizer.get_vob_size())
    for ln in nlp.next_line_from_files(data_files):
      sample = eval(ln)
      word_ids = tokenizer.tokenize(sample['text'], param.max_seq_len)
      mask_size = len(word_ids)
      word_ids.extend([tokenizer._pad_idx] * (param.max_seq_len - len(word_ids)))
      labels = [0] * param.max_seq_len
      for key, value_list in sample["tags"].items():
        for value in value_list:
          for pos in range(value[0], value[1] + 1):
            if pos >= param.max_seq_len:
              break
            if pos == value[0]:
              labels[pos] = param.tag_list.index(key) * 2 - 1
            else:
              labels[pos] = labels[value[0]] + 1
      # print(sample['text'])
      yield word_ids[:param.max_seq_len], labels[:param.max_seq_len], min(mask_size, param.max_seq_len)

  data = list(data_generator())
  pickle.dump(data, open(out_file, "wb"))

if __name__ == '__main__':
  param = Param()
  process(
    [f"{param.path_data}/train.data"],
    os.path.join(param.path_feat, "train.pydict"), param
  )
