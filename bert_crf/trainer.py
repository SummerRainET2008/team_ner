# coding: utf8
# author: Tian Xia (SummerRainET2008@gmail.com)

'''
#todo:
1. to support POS-tagging, external features.
2. in the preprocessing, Arabic should be processed.
   e.g., 1634--> one token。
'''

from pa_nlp.tf_1x.nlp_tf import *
from bert_crf._model import _Model
from bert_crf.data import *
from pa_nlp import chinese


#todo: summer: follow our team coding style.
class Trainer(object):
    def train(self, param):
        self.param = param
        assert param["evaluate_frequency"] % 100 == 0

        os.environ["CUDA_VISIBLE_DEVICES"] = str(param["GPU"])

        self._create_model()
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())

        param = self.param
        # tag_list: guarantee to be pre-sorted, "O" is the first one
        train_data = DataSet(data_file=param["train_file"],
                             tag_list=param["tag_list"],
                             max_seq_len=param["max_seq_length"],
                             vob_file=param["vob_file"],
                             )
        batch_iter = train_data.create_batch_iter(batch_size=param["batch_size"],
                                                  epoch_num=param["epoch_num"],
                                                  shuffle=True)

        if is_none_or_empty(param["vali_file"]):
            vali_data = None
        else:
            vali_data = DataSet(data_file=param["vali_file"],
                                tag_list=param["tag_list"],
                                max_seq_len=param["max_seq_length"],
                                vob_file=param["vob_file"]
                                )

        self._best_vali_accuracy = None

        model_dir = param["model_dir"]
        execute_cmd(f"rm -rf {model_dir}; mkdir {model_dir}")
        self._model_prefix = os.path.join(model_dir, "iter")
        self._saver = tf.train.Saver(max_to_keep=5)
        param_file = os.path.join(model_dir, "param.pydict")
        write_pydict_file([param], param_file)

        display_freq = 1
        accum_loss = 0.
        last_display_time = time.time()
        accum_run_time = 0
        for step, [input_ids, input_mask, segment_ids, y_batch, seq_len] in enumerate(batch_iter):
            start_time = time.time()
            print(y_batch)
            loss, accuracy = self._go_a_step(input_ids,
                                             input_mask,
                                             segment_ids,
                                             y_batch,
                                             seq_len,
                                             param["dropout_keep_prob"])
            duration = time.time() - start_time

            accum_loss += loss
            accum_run_time += duration
            if (step + 1) % display_freq == 0:
                accum_time = time.time() - last_display_time
                avg_loss = accum_loss / display_freq

                print(f"step: {step + 1}, avg loss: {avg_loss:.4f}, "
                      f"accuracy: {accuracy:.4f}, "
                      f"time: {accum_time:.4} secs, "
                      f"data reading time: {accum_time - accum_run_time:.4} sec.")

                accum_loss = 0.
                last_display_time = time.time()
                accum_run_time = 0

            if (step + 1) % param["evaluate_frequency"] == 0:
                if vali_data is not None:
                    self._validate(vali_data, step)
                else:
                    self._save_model(step)

        if vali_data is None:
            self._save_model(step)
        else:
            self._validate(vali_data, step)

    def _save_model(self, step):
        self._saver.save(self._sess, self._model_prefix, step)

    def _validate(self, vali_data, step):
        vali_iter = vali_data.create_batch_iter(128, 1, False)
        correct = 0
        for input_ids, input_mask, segment_ids, batch_y, seq_len in vali_iter:
            _, _, accuracy = Trainer.predict(self._sess, self.param, input_ids, input_mask, segment_ids, seq_len, batch_y)
            correct += accuracy * len(batch_y)

        accuracy = correct / vali_data.size()
        if self._best_vali_accuracy is None or accuracy > self._best_vali_accuracy:
            self._best_vali_accuracy = accuracy
            self._save_model(step)

        print(f"evaluation: accuracy: {accuracy:.4f} "
              f"best: {self._best_vali_accuracy:.4f}\n")

    @staticmethod
    def translate(word_list: list, predict_seq: list, tag_list: list):
        '''
    :param word_list: must be normalized word list.
    :return: {"PERSON": [str1, str2], "ORG": [str1, str2]}
    '''
        buffer = []
        for idx, label in enumerate(predict_seq):
            if idx == len(word_list):
                break

            if label == 0:
                continue

            elif label % 2 == 1:
                buffer.append([tag_list[(label + 1) // 2]])
                buffer[-1].append(word_list[idx])

            elif label % 2 == 0:
                buffer[-1].append(word_list[idx])

        ret = defaultdict(list)
        for match in buffer:
            ret[match[0]].append(chinese.join_ch_en_tokens(match[1:]))

        return ret

    @staticmethod
    def predict(sess, param, input_ids, input_mask, segment_ids, seq_len, batch_y=None):
      if not batch_y:
        batch_y = [[0] * param["max_seq_length"]] * param["batch_size"]

      graph = sess.graph
      result = sess.run(
        [
            graph.get_tensor_by_name("opt_seq:0"),
            graph.get_tensor_by_name("opt_seq_prob:0"),
            graph.get_tensor_by_name("accuracy:0"),
        ],
        feed_dict={
          graph.get_tensor_by_name("dropout:0"): 1,
          graph.get_tensor_by_name("input_y:0"): batch_y,
          graph.get_tensor_by_name("input_ids:0"): input_ids,
          graph.get_tensor_by_name("input_mask:0"): input_mask,
          graph.get_tensor_by_name("segment_ids:0"): segment_ids,
          graph.get_tensor_by_name("seq_len:0"): seq_len,
        }
      )

      return result

    def _go_a_step(self, input_ids, input_mask, segment_ids, y_batch, seq_len, dropout_keep_prob):
        result = self._sess.run(
            [
                self._train_optimizer,
                self._model.loss,
                self._model.accuracy,
                self._model.opt_seq,
                self._model.opt_seq_prob,
                self._model.init_probs,
                self._model.trans_probs,
                self._model.states2tags,
            ],
            feed_dict={
                self._model.input_y: y_batch,
                self._model.seq_len: seq_len,
                self._model.dropout_keep_prob: dropout_keep_prob,
                self._model.bert_model.input_ids: input_ids,
                self._model.bert_model.input_padding_mask: input_mask,
                self._model.bert_model.input_segment_ids: segment_ids,
            }
        )
        print(result[3])
        return result[1], result[2]

    def _create_model(self):
        self._model = _Model(
            max_seq_len=self.param["max_seq_length"],
            tag_size=2 * len(self.param["tag_list"]) - 1,
            is_training=True,
            bert_config_file=self.param["bert_config_file"],
            bert_init_ckpt=self.param["bert_init_ckpt"],
        )

        optimizer = tf.train.AdamOptimizer(self.param["learning_rate"])
        grads_and_vars = optimizer.compute_gradients(self._model.loss)
        self._train_optimizer = optimizer.apply_gradients(grads_and_vars)
