from bi_lstm_crf_torch.param import Param
from bi_lstm_crf_torch.model import BiLSTM_CRF
from bi_lstm_crf_torch.make_features import Tokenizer
from my_common import *

param = Param()
# param.verify()

model = BiLSTM_CRF(max_seq_len=param.max_seq_len,
                   tag_size=len(param.tag_list)*2-1,
                   vob_size=param.vob_size,
                   embedding_size=param.embedding_size,
                   LSTM_layer_num=param.LSTM_layer_num,
                   dropout=param.dropout)

model = load_specific_model(model, './work.bi_lstm_crf_torch/model/model_8.pt')

test_text = "My name is Jingwen Huang."
tokenizer = Tokenizer.get_instance()
word_ids = tokenizer.tokenize(sentence=test_text, max_len=param.max_seq_len)

tag_list = model(torch.tensor([word_ids]))
tag_list = [[str(item) for item in seq] for seq in tag_list]
print(index_to_tag(tag_list, param.tag_list))
