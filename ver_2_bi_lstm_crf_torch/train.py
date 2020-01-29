import torch
import time
import optparse
from ver_2_bi_lstm_crf_torch import *
from ver_2_bi_lstm_crf_torch.param import Param
from ver_2_bi_lstm_crf_torch.model import BiLSTM_CRF
from ver_2_bi_lstm_crf_torch.dataset import get_batch_data
from ver_2_bi_lstm_crf_torch.make_features import Tokenizer
from pa_nlp.nlp import Logger
from pa_nlp.pytorch.estimator.train import TrainerBase
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

class Trainer(TrainerBase):
  def __init__(self):
    param = Param()
    param.verify()
    self.param = param

    model = BiLSTM_CRF(max_seq_len=param.max_seq_len,
                       tag_size=len(param.tag_list)*2-1,
                       vob_size=param.vob_size,
                       embedding_size=param.embedding_size,
                       LSTM_layer_num=param.LSTM_layer_num,
                       dropout=param.dropout)

    optimizer_parameters = [
      {
        'params': [
          p for n, p in model.named_parameters()
        ],
        'lr': param.lr,
      },
    ]

    optimizer = getattr(torch.optim, param.optimizer_name)(
      optimizer_parameters
    )

    super(Trainer, self).__init__(
      param, model,
      get_batch_data(param.train_files, param.epoch_num, True),
      optimizer
    )

  def evaluate_file(self, data_file: str):
    start_time = time.time()
    all_true_labels = []
    all_pred_labels = []
    for _, batch in get_batch_data([data_file], 1, False):
      batch = [e.to(self._device) for e in batch]
      b_word_ids, b_labels = batch
      opt_seq = self.predict(b_word_ids)
      all_pred_labels.extend(opt_seq)
      all_true_labels.extend(b_labels.tolist())
      all_pred_labels = [[str(item) for item in seq] for seq in all_pred_labels]
      all_true_labels = [[str(item) for item in seq] for seq in all_true_labels]
      all_pred_labels = index_to_tag(all_pred_labels, self.param.tag_list)
      all_true_labels = index_to_tag(all_true_labels, self.param.tag_list)
      print(all_pred_labels)
      print(all_true_labels)

    f_score = f1_score(all_true_labels, all_pred_labels)
    avg_accuracy = cal_accuracy(all_pred_labels, all_true_labels)
    precision = precision_score(all_true_labels, all_pred_labels)
    total_time = time.time() - start_time
    avg_time = total_time / (len(all_true_labels) + 1e-6)
    Logger.info(
      f"eval[{self._run_sample_num}]: "
      f"file={data_file} weighted_f={f_score:.4f} accuracy={avg_accuracy:.4f} precision={precision:.4f} "
      f"total_time={total_time:.4f} secs avg_time={avg_time:.4f} sec/sample "
    )

    print(classification_report(all_true_labels, all_pred_labels))

    return -f_score


  def _train_one_batch(self, b_word_ids, b_labels):
    self._model.train()
    self._optimizer.zero_grad()
    loss = self._model.neg_log_likelihood(b_word_ids, b_labels)

    return loss

  def predict(self, b_sent):
    pred_labels = self._model(b_sent)
    return pred_labels


def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  parser.add_option("--debug_level", type=int, default=1)
  (options, args) = parser.parse_args()
  param = Param()

  Logger.set_level(options.debug_level)

  trainer = Trainer()
  trainer.train()
  new_trainer = Trainer()
  test_text = "My name is Jingwen Huang."
  tokenizer = Tokenizer.get_instance()
  word_ids = tokenizer.tokenize(sentence=test_text, max_len=param.max_seq_len)
  tag_list = new_trainer.predict(torch.tensor([word_ids]))
  tag_list = [[str(item) for item in seq] for seq in tag_list]
  print(index_to_tag(tag_list, param.tag_list))


if __name__ == "__main__":
  main()
