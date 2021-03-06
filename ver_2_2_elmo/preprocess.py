# coding: utf8

import argparse
import torch
import os
import logging
import ver_2_2_elmo.constants as constants

def read_instances_from_file(inst_file, max_sent_len, keep_case):
  '''
  Convert file into word seq lists and vocab
  Returns: [n_line,l_seq]
  '''

  word_insts = []
  trimmed_sent_count = 0
  with open(inst_file) as f:
    for sent in f:
      if not keep_case:
        sent = sent.lower()
      words = sent.split()
      if len(words) > max_sent_len:
        trimmed_sent_count += 1
      word_inst = words[:max_sent_len]

      if word_inst:
        word_insts += [word_inst]
      else:
        word_insts += [None]

  print('[Info] Get {} instances from {}'.format(
    len(word_insts), inst_file
  ))

  if trimmed_sent_count > 0:
    print(
      '[Warning] {} instances are trimmed to the max sentence length {}.'.format(
        trimmed_sent_count, max_sent_len
      ))

  return word_insts


def get_char_input(word_insts, max_char_len):
  '''
  Split char of returns of def read_instances_from_file
  Args:
    word_insts: [n_line,l_seq]
    max_char_len: int 15
  Returns: list of list of list, [n_line,l_seq,l_char]
  '''

  tensor_list = []
  for sequence in word_insts:
    mtrx = []
    for row_idx in range(len(sequence)):
      vec = []
      for col_idx, content in enumerate(sequence[row_idx]):
        vec.append(content)
      vec = vec[:max_char_len]
      mtrx.append(vec)
    tensor_list.append(mtrx)
  return tensor_list


def build_char_idx(char_insts, min_char_count):
  '''
  Args:
    char_insts: [n_line,l_seq,l_word]
  '''

  full_chars = set(c for sent in char_insts for w in sent for c in w)
  print('full_chars:', full_chars)
  print('[Info] Original character number  =', len(full_chars))
  char2idx = {
    constants.PAD_WORD: constants.PAD, constants.UNK_WORD: constants.UNK
  }
  char_count = {c: 0 for c in full_chars}
  for sent in char_insts:
    for word in sent:
      for i in range(1, len(word)):
        char_count[word[i]] += 1
  ignored_char_count = 0
  for word, count in char_count.items():
    if word not in char2idx:
      if count >= min_char_count:
        char2idx[word] = len(char2idx)
      else:
        ignored_char_count += 1

  print('[Info] Trimmed Chracter size = {},'.format(len(char2idx)),
        'each with minimum occurrence = {}'.format(min_char_count))
  print("[Info] Ignored word count = {}".format(ignored_char_count))
  return char2idx,


def build_vocab_idx(word_insts, min_word_count, tgt_tag=False):
  '''
  Trim vocab by number of occurence
  Args:
    word_insts: tuple of lists, every list is words in a sentence (src / tgt)
    min_word_count: ignore words appearing less than min_word_count times
  Returns: vocabulary
  '''

  if tgt_tag:
    full_vocab = set(
      w for sent in word_insts for i, w in enumerate(sent) if i != 0
    )
    full_intent = set(
      w for sent in word_insts for i, w in enumerate(sent) if i == 0
    )
    print('[Info] Original Vocabulary size =', len(full_vocab))
    print('[Info] Original Intent size =', len(full_intent))
  else:
    full_vocab = set(w for sent in word_insts for i, w in enumerate(sent))
    print('[Info] Original Vocabulary size =', len(full_vocab))

  word2idx = {constants.BOS_WORD: constants.BOS,
              constants.EOS_WORD: constants.EOS,
              constants.PAD_WORD: constants.PAD,
              constants.UNK_WORD: constants.UNK}
  intent2idx = {}

  # word_count: {word: number}
  word_count = {w: 0 for w in full_vocab}
  for sent in word_insts:
    for i in range(1, len(sent)):
      word_count[sent[i]] += 1

  ignored_word_count = 0
  if not tgt_tag:
    for word, count in word_count.items():
      if word not in word2idx:
        if count >= min_word_count:
          word2idx[word] = len(word2idx)
        else:
          ignored_word_count += 1
  if tgt_tag:
    for word, count in word_count.items():
      if word not in word2idx:
        word2idx[word] = len(word2idx)
    for intent in full_intent:
      intent2idx[intent] = len(intent2idx) + len(word2idx)

  print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
        'each with minimum occurrence = {}'.format(min_word_count))
  print("[Info] Ignored word count = {}".format(ignored_word_count))
  if tgt_tag:
    return word2idx, intent2idx
  else:
    return word2idx,


def convert_char_instance_to_idx_seq(char_insts, char2idx):
  ''' Mapping words to idx sequence. '''
  return [[[char2idx.get(c, constants.UNK) for c in w] for w in s]
          for s in char_insts]


def convert_instance_to_idx_seq(word_insts, word2idx,
                                intent2idx=None, is_src=True):
  ''' Mapping words to idx sequence. '''
  if is_src:
    return [[word2idx.get(w, constants.UNK) for w in s] for s in word_insts]
  else:
    return [[word2idx.get(w, intent2idx.get(w, constants.UNK)) for w in s]
            for s in word_insts]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '-train_src', default='data/snips/train/intent_seq.in', required=False)
  parser.add_argument(
    '-train_tgt', default='data/snips/train/intent_seq.out', required=False)
  parser.add_argument(
    '-valid_src', default='data/snips/valid/intent_seq.in', required=False)
  parser.add_argument(
    '-valid_tgt', default='data/snips/valid/intent_seq.out', required=False)
  parser.add_argument(
    '-test_src', default='data/snips/test/intent_seq.in', required=False)
  parser.add_argument(
    '-test_tgt', default='data/snips/test/intent_seq.out', required=False)

  parser.add_argument('-save_data', default='data/snips.pt', required=False)
  parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=60)
  parser.add_argument('-min_word_count', type=int, default=2)
  parser.add_argument('-max_char_len', type=int, default=15)
  parser.add_argument('-keep_case', default=True)
  parser.add_argument('-share_vocab', default=False)
  parser.add_argument('-vocab', default=None)

  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

  logger = logging.getLogger()  # this gets the root logger
  logger.setLevel(logging.WARNING)

  torch.manual_seed(1)
  torch.cuda.manual_seed(1)

  opt = parser.parse_args()
  opt.max_token_seq_len = opt.max_word_seq_len
  train_src_word_insts = read_instances_from_file(
    opt.train_src, opt.max_word_seq_len, opt.keep_case)
  train_tgt_word_insts = read_instances_from_file(
    opt.train_tgt, opt.max_word_seq_len, opt.keep_case)
  train_src_char_insts = get_char_input(train_src_word_insts, opt.max_char_len)

  if len(train_src_word_insts) != len(train_tgt_word_insts):
    print('[Warning] The training instance count is not equal.')
    min_inst_count = min(len(train_src_word_insts), len(train_tgt_word_insts))
    train_src_word_insts = train_src_word_insts[:min_inst_count]
    train_tgt_word_insts = train_tgt_word_insts[:min_inst_count]

  # - Remove empty instances
  train_src_word_insts, train_tgt_word_insts = list(
    zip(*[(s, t) for s, t in zip(train_src_word_insts, train_tgt_word_insts) if
          s and t])
  )

  for i in range(len(train_src_word_insts)):
    assert len(train_src_word_insts[i]) == len(train_tgt_word_insts[i])

  valid_src_word_insts = read_instances_from_file(
    opt.valid_src, opt.max_word_seq_len, opt.keep_case
  )
  valid_tgt_word_insts = read_instances_from_file(
    opt.valid_tgt, opt.max_word_seq_len, opt.keep_case
  )
  valid_src_char_insts = get_char_input(valid_src_word_insts, opt.max_char_len)

  if len(valid_src_word_insts) != len(valid_tgt_word_insts):
    print('[Warning] The validation instance count is not equal.')
    min_inst_count = min(len(valid_src_word_insts), len(valid_tgt_word_insts))
    valid_src_word_insts = valid_src_word_insts[:min_inst_count]
    valid_tgt_word_insts = valid_tgt_word_insts[:min_inst_count]

  # - Remove empty instances
  valid_src_word_insts, valid_tgt_word_insts = list(
    zip(*[(s, t) for s, t in zip(valid_src_word_insts, valid_tgt_word_insts) if
          s and t])
  )  # zip(*zipped) functions as unzip..

  test_src_word_insts = read_instances_from_file(
    opt.test_src, opt.max_word_seq_len, opt.keep_case
  )
  test_tgt_word_insts = read_instances_from_file(
    opt.test_tgt, opt.max_word_seq_len, opt.keep_case
  )
  test_src_char_insts = get_char_input(test_src_word_insts, opt.max_char_len)

  if len(test_src_word_insts) != len(test_tgt_word_insts):
    print('[Warning] The validation instance count is not equal.')
    min_inst_count = min(len(test_src_word_insts), len(test_tgt_word_insts))
    test_src_word_insts = test_src_word_insts[:min_inst_count]
    test_tgt_word_insts = test_tgt_word_insts[:min_inst_count]

  # - Remove empty instances
  test_src_word_insts, test_tgt_word_insts = list(
    zip(*[(s, t) for s, t in zip(test_src_word_insts, test_tgt_word_insts) if
          s and t])
  )

  # Build vocabulary
  if opt.vocab:
    predefined_data = torch.load(opt.vocab)
    assert 'dict' in predefined_data
    print('[Info] Pre-defined vocabulary found.')
    src_word2idx = predefined_data['dict']['src']
    tgt_word2idx = predefined_data['dict']['tgt']
    intent2idx = predefined_data['dict']['intent']
    char2idx, *_ = build_char_idx(train_src_char_insts, opt.min_word_count)
  else:
    print('[Info] Build vocabulary for source.')
    src_word2idx, *_ = build_vocab_idx(
      train_src_word_insts, opt.min_word_count, tgt_tag=False)
    print('[Info] Build vocabulary for target.')
    tgt_word2idx, intent2idx = build_vocab_idx(
      train_tgt_word_insts, opt.min_word_count, tgt_tag=True)
    char2idx, *_ = build_char_idx(train_src_char_insts, opt.min_word_count)

  print('[Info] Convert source word instances into sequences of word index.')
  train_src_insts = convert_instance_to_idx_seq(
    train_src_word_insts, src_word2idx, is_src=True)
  valid_src_insts = convert_instance_to_idx_seq(
    valid_src_word_insts, src_word2idx, is_src=True)
  test_src_insts = convert_instance_to_idx_seq(
    test_src_word_insts, src_word2idx, is_src=True)

  print('[Info] Convert target word instances into sequences of word index.')
  train_tgt_insts = convert_instance_to_idx_seq(
    train_tgt_word_insts, tgt_word2idx, intent2idx=intent2idx, is_src=False
  )
  valid_tgt_insts = convert_instance_to_idx_seq(
    valid_tgt_word_insts, tgt_word2idx, intent2idx=intent2idx, is_src=False
  )
  test_tgt_insts = convert_instance_to_idx_seq(
    test_tgt_word_insts, tgt_word2idx, intent2idx=intent2idx, is_src=False
  )

  embeddings = None
  data = {'settings': opt, 'dict': {'src': src_word2idx, 'tgt': tgt_word2idx,
                                    'intent': intent2idx, },
          'train': {'src': train_src_insts, 'src_char': train_src_word_insts,
                    'tgt': train_tgt_insts},
          'valid': {'src': valid_src_insts, 'src_char': valid_src_word_insts,
                    'tgt': valid_tgt_insts},
          'test': {'src': test_src_insts, 'src_char': test_src_word_insts,
                   'tgt': test_tgt_insts},
          'embeddings': embeddings}

  print('[Info] Dumping the processed data to pickle file', opt.save_data)
  torch.save(data, opt.save_data)
  print('[Info] Finish.')


if __name__ == '__main__':
  main()
