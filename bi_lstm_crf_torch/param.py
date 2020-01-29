import os
from pa_nlp.pytorch import *
from pa_nlp.pytorch.estimator.param import ParamBase

TAG_LIST = ['ORG', 'TIME', 'FAC', 'LANGUAGE', 'GPE', 'WORK_OF_ART', 'CARDINAL', 'QUANTITY', 'DATE', 'PRODUCT', 'NORP',
            'LOC', 'LAW', 'ORDINAL', 'PERSON', 'EVENT', 'O', 'PERCENT', 'MONEY']

class Param(ParamBase):
  def __init__(self):
    super(Param, self).__init__("bi_lstm_crf_torch")

    self.train_files = [
      f"{self.path_feat}/train.pydict"
    ]

    self.vali_files = [
      f"{self.path_feat}/train.pydict"
    ]

    self.test_files = [
      f"{self.path_feat}/train.pydict"
    ]

    self.pretrained_model = os.path.expanduser(
      "../../roberta/roberta.base/",
    )

    self.optimizer_name = "Adam"
    self.lr = 0.001
    self.l2 = 0
    self.epoch_num = 10
    self.single_GPU_batch_size = 128
    self.iter_num_update_optimizer = 1
    self.max_seq_len = 32
    self.eval_gap_instance_num = 5

    self.tag_list = ["O"] + sorted(set(TAG_LIST) - set("O"))
    self.embedding_size = 128
    self.RNN_type = "lstm"
    self.GPU = -1
    self.gpus = []
    self.dropout = 0.5
    self.vob_size = 60000
    self.LSTM_layer_num = 2
    self.incremental_train = False
    self.use_polynormial_decay = False
    self.model_kept_num = 30
    self.train_sample_num = 3



if __name__ == '__main__':
  param = Param()
  print(param.optimizer_name)