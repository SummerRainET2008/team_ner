#coding: utf8
#author: Tian Xia (SummerRainET2008@gmail.com)

from pa_nlp.tf_1x import nlp_tf as TF
import tensorflow as tf
from pa_nlp.common import *
from pa_nlp.tf_1x.bert.bert_model import *

class _Model(object):
  def __init__(self,
               max_seq_len,
               tag_size,  # ['O', 'B_TAG1', 'I_TAG2' ...]
               is_training,
               bert_config_file,
               bert_init_ckpt
              ):

    self.input_y = tf.placeholder(tf.int32, [None, max_seq_len],
                                  name="input_y")
    self.seq_len = tf.placeholder(tf.int32, [None],
                                  name="seq_len")
    self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout")

    print(f"name(input_y):{self.input_y.name}")
    print(f"name(dropout_keep_prob):{self.dropout_keep_prob.name}")
    
    self._tag_size = tag_size

    init_probs = tf.get_variable("raw_init_probs", [tag_size], tf.float32)
    self.init_probs = self._norm(init_probs, 0, "init_probs")
    
    trans_probs = tf.get_variable("raw_trans_probs",
                                  [tag_size, tag_size], tf.float32)
    self.trans_probs = self._norm(trans_probs, 1, "trans_probs")

    self.bert_model = BertModel(bert_config_file=bert_config_file,
                                max_seq_length=max_seq_len,
                                bert_init_ckpt=bert_init_ckpt,
                                is_training=is_training)

    bert_states = self.bert_model.get_layer_output()[-1]
    bert_states = tf.unstack(bert_states, axis=1)

    bert_states = tf.unstack(tf.nn.dropout(bert_states,
                                              self.dropout_keep_prob))
    self.states2tags = [self._observation_tag_probs(state, pos)
                        for pos, state in enumerate(bert_states)]
    
    self._calc_loss_tf()
    self._search_tf()
    
  def _norm(self, tensor, axis, name):
    return tf.log(tf.nn.softmax(tensor, axis=axis), name)
    
  def _observation_tag_probs(self, bi_LSTM_state, pos):
    with tf.variable_scope("observation_tag_probs", reuse=pos > 0):
      return self._norm(tf.layers.dense(bi_LSTM_state, self._tag_size),
                        1, f"{pos}")


  def _calc_loss_tf(self):
    state_scores = tf.stack(self.states2tags)
    state_scores = tf.transpose(state_scores, [1, 0, 2])
    print(f'The size of state_scores is {state_scores.get_shape().as_list()}')
    log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(inputs=state_scores,
                                                          tag_indices=self.input_y,
                                                          transition_params=self.trans_probs,
                                                          sequence_lengths=self.seq_len
                                                          )
    self.loss = tf.identity(tf.reduce_mean(-log_likelihood), name="loss")
    print(f"name(._loss):{self.loss.name}")


  def _search_tf(self):
    state_scores = tf.stack(self.states2tags)
    state_scores = tf.transpose(state_scores, [1, 0, 2])
    opt_seq, opt_seq_prob = tf.contrib.crf.crf_decode(potentials=state_scores,
                                                      transition_params=self.trans_probs,
                                                      sequence_length=self.seq_len
                                                      )
    self.opt_seq = tf.identity(opt_seq, "opt_seq")
    self.opt_seq_prob = tf.identity(opt_seq_prob, "opt_seq_prob")
    print(f"name(opt_seq):{self.opt_seq.name}")
    print(f"name(opt_seq_prob):{self.opt_seq_prob.name}")

    self.accuracy = tf.identity(
      TF.accuracy(self.opt_seq, self.input_y), "accuracy"
    )
    print(f"name(accuracy):{self.accuracy.name}")
  
