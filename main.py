# //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

from lib.preprocessing.pheme.preprocess_impl import PreProcessImpl
from lib.read_datasets.pheme.read_pheme_ds import read_pheme_ds
from lib.training_modules.bilstm.bilstm_impl import BiLstmImpl

df = read_pheme_ds()

pre_process = PreProcessImpl(df=df)
preprocessed_df = pre_process.get_preprocessed_dataframe()

bi_lstm = BiLstmImpl()
x_train, x_test, y_train, y_test = bi_lstm.data_reshape(df=preprocessed_df)
bi_lstm.run_bi_lstm(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, max_len=64)

import torch
from transformers import AutoTokenizer, AutoModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)

# bert_preprocess = KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
# text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
# preprocessor = hub.KerasLayer(
#     "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
# encoder_inputs = preprocessor(text_input)
# import torch

# from transformers import BertTokenizer, BertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
# example_text = '[CLS] ' + "The man I've seen." + ' [SEP] ' + 'He bough a gallon of milk.' + ' [SEP] '
# print('sample text is : ' + example_text)
# tokenized_text = tokenizer.tokenize(text=example_text)
#
# indexed_tokens = tokenizer.convert_tokens_to_ids(tokens=tokenized_text)
#
# for tup in zip(tokenized_text, indexed_tokens):
#     print('{:<12}{:>6,}'.format(tup[0], tup[1]))
#
# segments_ids = [1] * len(tokenized_text)
#
# print('segments_ids are ' + str(segments_ids))
# tokens_tensor = torch.tensor([indexed_tokens])
# print(tokens_tensor)
# print('tokens_tensor are ' + str(tokens_tensor))
#
# segments_tensors = torch.tensor([segments_ids])
# print(segments_tensors)
# print('segments_tensors are ' + str(segments_tensors))
#
# model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
#
#
# model.eval()
# with torch.no_grad():
#     outputs = model(tokens_tensor, segments_tensors)
#     hidden_states = outputs[2]
#     print(outputs[0])
#     print(outputs[1])
#     print(outputs[2])
#
#     print(hidden_states)
