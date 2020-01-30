from pa_nlp.tf_1x.nlp_tf import *
from ver_1_2_bert_crf.data import *
from ver_1_2_bert_crf.trainer import Trainer
from pa_nlp.tf_1x.bert.bert_tokenizer import *
from ver_1_2_bert_crf._model import _Model

class Predictor(object):
  def __init__(self, model_path):
    ''' We only load the best model in {model_path}
    '''
    def extract_id(model_file):
      return int(re.findall(r"iter-(.*?).index", model_file)[0])

    param_file = os.path.join(model_path, "param.pydict")
    self.param = list(read_pydict_file(param_file))[0]

    names = [extract_id(name) for name in os.listdir(model_path)
             if name.endswith(".index")]
    best_iter = max(names)
    model_prefix = f"{model_path}/iter-{best_iter}"
    print(f"loading model: '{model_prefix}'")
    
    graph = tf.Graph()
    with graph.as_default():
      self._model = _Model(max_seq_len=self.param["max_seq_length"],
            tag_size=2 * len(self.param["tag_list"]) - 1,
            is_training=False,
            bert_init_ckpt=self.param["bert_init_ckpt"],
            bert_config_file=self.param["bert_config_file"],
            )

    self._sess = tf.Session(graph=graph)
    with self._sess.as_default():
      with graph.as_default():
        tf.train.Saver().restore(self._sess, model_prefix)
    
  def translate(self, word_list: list, predict_seq: list):
    return Trainer.translate(word_list, predict_seq, self.param["tag_list"])

  def predict_dataset(self, file_name):
    data = DataSet(data_file=file_name,
                   tag_list=self.param["tag_list"],
                   max_seq_len=self.param["max_seq_length"],
                   vob_file=self.param["vob_file"])
    data_iter = data.create_batch_iter(batch_size=self.param["batch_size"],
                                       epoch_num=1,
                                       shuffle=False)
    fou = open(file_name.replace(".pydict", ".pred.pydict"), "w")
    correct = 0.
    for input_ids, input_mask, segment_ids, batch_y, seq_len in data_iter:
      preds, probs, accuracy = self.predict(input_ids, input_mask, segment_ids, seq_len, batch_y)
      correct += accuracy * len(batch_y)
      
      for idx, y in enumerate(batch_y):
        pred = {
          "predicted_tags": list(preds[idx]),
          "prob": probs[idx],
        }
        print(pred, file=fou)
    fou.close()
    
    accuracy = correct / data.size()
    print(f"Test: '{file_name}': {accuracy:.4f}")
    
  def predict_one_sample(self, sentence: str):
    '''
    :param word_list: must be processed by normalization.
    :return: [translation, prob]
    '''
    T = BertTokenizer(vocab_file=self.param["vob_file"])
    tokens, input_ids, input_mask, segment_ids = T.parse_single_sentence(
      text_a=sentence, max_seq_length=self.param["max_seq_length"]
    )
    input_ids = input_ids[:self.param["max_seq_length"]]
    input_mask = input_mask[:self.param["max_seq_length"]]
    segment_ids = input_mask[:self.param["max_seq_length"]]
    seq, prob, _ = self.predict([input_ids], [input_mask], [segment_ids], [len(tokens)], None)
    seq, prob = seq[0], prob[0]
    
    # tran = self.translate(word_list, seq)
    
    return seq, prob
    
  def predict(self, input_ids, input_mask, segment_ids, seq_len, batch_y):
    '''
    :param batch_x: must be of the length used in training.
    :param batch_y: could be None
    :return: [seq, seq_prob, accuracy]
    '''
    return Trainer.predict(self._sess, self.param, input_ids, input_mask, segment_ids, seq_len, batch_y)

