import sys
sys.path.append('../my-tool-box/insight_nlp/')
from pa_nlp.nlp import Logger
import numpy as np
import torch


def cal_accuracy(prediction, label):
  prediction = np.array(prediction)
  label = np.array(label)
  hit = sum(prediction==label)
  while type(hit) is not np.int64:
    hit = sum(hit)
  miss = sum(prediction!=label)
  while type(miss) is not np.int64:
    miss = sum(miss)
  return hit/(hit+miss)


def index_to_tag(label: list, tag_list):
  index_2_tag = {}
  for index, tag in enumerate(tag_list):
    if tag == 'O':
      index_2_tag['0'] = tag
    else:
      index_2_tag[str(index*2-1)] = "B-" + tag
      index_2_tag[str(index*2)] = "I-" + tag

  new_tag_list = []
  for seq in label:
    seq_tag_list = []
    for tag in seq:
      seq_tag_list.append(index_2_tag[tag])
    new_tag_list.append(seq_tag_list)

  return new_tag_list


def load_specific_model(model, model_file_path):
  checked_data = torch.load(model_file_path)

  state_dict = checked_data[3]
  model.load_state_dict(state_dict)
  Logger.info(f"Model load succeeds: {model_file_path}")

  return model


if __name__ == '__main__':
  test_1 = torch.tensor([[1,2,3], [4,5,6]])
  test_2 = torch.tensor([[1,2,2], [4,5,6]])
  print(cal_accuracy(test_1, test_2))
  tag_list = ['ORG', 'TIME', 'FAC', 'LANGUAGE', 'GPE', 'WORK_OF_ART', 'CARDINAL', 'QUANTITY', 'DATE', 'PRODUCT', 'NORP',
              'LOC', 'LAW', 'ORDINAL', 'PERSON', 'EVENT', 'O', 'PERCENT', 'MONEY']
  tag_list = ["O"] + sorted(set(tag_list) - set("O"))
  print(index_to_tag([['1', '2', '0', '0'], ['0', '3', '4', '4']], tag_list))
