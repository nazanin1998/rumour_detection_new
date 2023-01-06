from sklearn.model_selection import train_test_split

from lib.training_modules.bert.bert_model_name import get_bert_preprocess_model_name
from lib.training_modules.bert.preprocess.bert_preprocessing import BertPreprocessing, \
    get_categorical_binary_numeric_string_feature_names
import tensorflow as tf
import tensorflow_hub as hub

from lib.training_modules.bert.bert_configurations import preprocess_ignore_exc_str, \
    preprocess_seq_length, preprocess_batch_size, preprocess_buffer_size, bert_test_size, bert_val_size, bert_train_size
from lib.utils.log.logger import log_start_phase, log_end_phase, log_line, log_phase_desc


class BertPreprocessingImpl(BertPreprocessing):
    def __init__(
            self,
            bert_model_name='bert_en_uncased_L-12_H-768_A-12'):
        self.label_feature_name, self.categorical_feature_names, self.binary_feature_names, self.numeric_feature_names, self.str_feature_names = \
            get_categorical_binary_numeric_string_feature_names()

        self.bert_preprocess_model_name = get_bert_preprocess_model_name(bert_model_name=bert_model_name)

    def start(self, df):
        log_start_phase(2, 'BERT PREPROCESSING')
        preprocess_layers = []

        available_splits, ds_size, label_classes, x_train_tensor, x_val_tensor, x_test_tensor, \
        y_train_tensor, y_val_tensor, y_test_tensor = self.ds_statistics(
            df=df,
            label_feature_name=self.label_feature_name,
            categorical_feature_names=self.categorical_feature_names,
            binary_feature_names=self.binary_feature_names,
            numeric_feature_names=self.numeric_feature_names,
            str_feature_names=self.str_feature_names)

        inputs = self.make_input_for_all_ds_columns(
            df=df,
            str_feature_names=self.str_feature_names,
            binary_feature_names=self.binary_feature_names,
            categorical_feature_names=self.categorical_feature_names,
            ignore_exc_str=preprocess_ignore_exc_str)

        if not preprocess_ignore_exc_str:
            preprocess_layers = self.make_binary_feature_preprocess_layer(
                preprocess_layers=preprocess_layers,
                binary_feature_names=self.binary_feature_names,
                inputs=inputs)

            preprocess_layers = self.make_numeric_feature_preprocess_layer(
                preprocess_layers=preprocess_layers,
                numeric_feature_names=self.numeric_feature_names,
                numeric_features=df[self.numeric_feature_names],
                inputs=inputs)

        preprocess_layers, bert_pack = self.make_string_feature_preprocess_layer(
            inputs=inputs,
            preprocess_layers=preprocess_layers,
            str_feature_names=self.str_feature_names)

        preprocessor_model = self.make_preprocess_model(
            preprocess_layers=preprocess_layers,
            inputs=inputs,
            bert_pack=bert_pack)

        preprocessor_model.summary()

        train_tensor_dataset, val_tensor_dataset, test_tensor_dataset = self.preprocess_test_train_val_ds(
            preprocessor_model=preprocessor_model,
            x_train_tensor=x_train_tensor,
            x_val_tensor=x_val_tensor,
            x_test_tensor=x_test_tensor,
            y_train_tensor=y_train_tensor,
            y_val_tensor=y_val_tensor,
            y_test_tensor=y_test_tensor,
        )

        log_end_phase(2, 'BERT PREPROCESSING')
        log_line()

        return train_tensor_dataset, val_tensor_dataset, test_tensor_dataset, label_classes, x_train_tensor.shape[0], \
               x_val_tensor.shape[0], x_test_tensor.shape[0], preprocessor_model

    def preprocess_test_train_val_ds(self,
                                     preprocessor_model,
                                     x_train_tensor,
                                     x_val_tensor,
                                     x_test_tensor,
                                     y_train_tensor,
                                     y_val_tensor,
                                     y_test_tensor
                                     ):
        train_size = x_train_tensor.shape[0]
        val_size = x_val_tensor.shape[0]
        test_size = x_test_tensor.shape[0]

        with tf.device('/cpu:0'):
            preprocessed_x_train = preprocessor_model(x_train_tensor)
            preprocessed_x_val = preprocessor_model(x_val_tensor)
            preprocessed_x_test = preprocessor_model(x_test_tensor)

        train_tensor_dataset = self.make_tensor_ds_of_preprocessed_data(
            label_tensor=y_train_tensor,
            preprocessed_train_features=preprocessed_x_train,
            num_examples=train_size,
            is_training=True)

        val_tensor_dataset = self.make_tensor_ds_of_preprocessed_data(
            label_tensor=y_val_tensor,
            preprocessed_train_features=preprocessed_x_val,
            num_examples=val_size,
            is_training=False)

        test_tensor_dataset = self.make_tensor_ds_of_preprocessed_data(
            label_tensor=y_test_tensor,
            preprocessed_train_features=preprocessed_x_test,
            num_examples=test_size,
            is_training=False)

        return train_tensor_dataset, val_tensor_dataset, test_tensor_dataset

    def ds_statistics(
            self,
            df,
            label_feature_name,
            categorical_feature_names,
            binary_feature_names,
            numeric_feature_names,
            str_feature_names):
        x = df[str_feature_names]
        y = df[label_feature_name]

        x_train, x_val, x_test, y_train, y_val, y_test = self.train_val_test_split(
            x=x,
            y=y,
            test_size=bert_test_size,
            train_size=bert_train_size,
            val_size=bert_val_size)

        x_train_tensor = self.__convert_to_tensor(x_train, dtype=tf.string)
        x_val_tensor = self.__convert_to_tensor(x_val, dtype=tf.string)
        x_test_tensor = self.__convert_to_tensor(x_test, dtype=tf.string)
        y_train_tensor = self.__convert_to_tensor(y_train, dtype=tf.int64)
        y_val_tensor = self.__convert_to_tensor(y_val, dtype=tf.int64)
        y_test_tensor = self.__convert_to_tensor(y_test, dtype=tf.int64)

        available_splits = list(['train', 'validation', 'test'])
        label_classes = len(y.value_counts())
        ds_size = df.shape[0]

        log_phase_desc(f'PHEME DS (SIZE)   : {ds_size}')
        log_phase_desc(f'LABEL CLASSES     : {label_classes}')
        log_phase_desc(f'TRAINING FEATURE  : {str_feature_names}')
        log_phase_desc(f'LABEL FEATURE     : {label_feature_name}\n')
        log_phase_desc(f'TRAIN      (SIZE) : {x_train_tensor.shape} ({bert_train_size * 100}%)')
        log_phase_desc(f'VALIDATION (SIZE) : {x_val_tensor.shape} ({bert_val_size * 100}%)')
        log_phase_desc(f'TEST       (SIZE) : {x_test_tensor.shape} ({bert_test_size * 100}%)')

        return available_splits, ds_size, label_classes, x_train_tensor, x_val_tensor, x_test_tensor, \
               y_train_tensor, y_val_tensor, y_test_tensor

    def make_input_for_all_ds_columns(
            self,
            df,
            str_feature_names,
            categorical_feature_names,
            binary_feature_names,
            ignore_exc_str):
        inputs = {}
        for name, column in df.items():

            if ignore_exc_str and name in str_feature_names:
                d_type = tf.string
                inputs[name] = tf.keras.Input(shape=(), name=name, dtype=d_type)

            if ignore_exc_str:
                continue

            if (name in categorical_feature_names or
                    name in binary_feature_names):
                d_type = tf.int64
            else:
                d_type = tf.float32

            inputs[name] = tf.keras.Input(shape=(), name=name, dtype=d_type)

        return inputs

    def make_binary_feature_preprocess_layer(
            self,
            preprocess_layers,
            binary_feature_names,
            inputs):
        for name in binary_feature_names:
            inp = inputs[name]
            inp = inp[:, tf.newaxis]
            float_value = tf.cast(inp, tf.float32)
            preprocess_layers.append(float_value)

        return preprocess_layers

    @staticmethod
    def __make_normalizer():
        return tf.keras.layers.Normalization(axis=-1)

    @staticmethod
    def __stack_dict(inputs, fun=tf.stack):
        values = []
        for key in sorted(inputs.keys()):
            values.append(tf.cast(inputs[key], tf.float32))

        return fun(values, axis=-1)

    @staticmethod
    def __convert_to_tensor(
            feature,
            dtype=None):
        return tf.convert_to_tensor(feature, dtype=dtype)

    @staticmethod
    def train_val_test_split(x, y, train_size, val_size, test_size):
        x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=test_size)

        relative_train_size = train_size / (val_size + train_size)
        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val,
                                                          train_size=relative_train_size,
                                                          test_size=1 - relative_train_size)
        return x_train, x_val, x_test, y_train, y_val, y_test

    def make_numeric_feature_preprocess_layer(
            self,
            preprocess_layers,
            numeric_feature_names,
            numeric_features,
            inputs):
        normalizer = self.__make_normalizer()

        normalizer.adapt(self.__stack_dict(dict(numeric_features)))

        numeric_inputs = {}
        for name in numeric_feature_names:
            numeric_inputs[name] = inputs[name]

        numeric_inputs = self.__stack_dict(numeric_inputs)
        numeric_normalized = normalizer(numeric_inputs)

        preprocess_layers.append(numeric_normalized)
        return preprocess_layers

    def make_string_feature_preprocess_layer(
            self,
            inputs,
            preprocess_layers,
            str_feature_names):

        bert_preprocess = hub.load(self.bert_preprocess_model_name)

        for name, input_item in inputs.items():
            if name not in str_feature_names:
                continue

            tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name='tokenizer')

            preprocessed_item = tokenizer(input_item)

            r"""first approach"""
            # lookup = tf.keras.layers.StringLookup(vocabulary=np.unique(df[name]))
            # one_hot = tf.keras.layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())
            # one_hot = tf.keras.layers.Embedding(input_dim=lookup.vocabulary_size(), output_dim=None)
            # x = lookup(input_item)
            # preprocessed_item = one_hot(x)

            r"""second approach"""
            # text_vectorizer = tf.keras.layers.TextVectorization()
            # text_vectorizer.adapt(df[name])
            # preprocessed_item = text_vectorizer(input_item)

            preprocess_layers.append(preprocessed_item)

        bert_pack = bert_preprocess.bert_pack_inputs
        return preprocess_layers, bert_pack

    def make_preprocess_model(
            self,
            preprocess_layers,
            inputs,
            bert_pack):
        packer = hub.KerasLayer(bert_pack,
                                arguments=dict(seq_length=preprocess_seq_length),
                                name='packer')
        preprocessed_result = packer(preprocess_layers)
        # preprocessed_result = tf.concat(preprocess_layers, axis=-1)

        preprocessor = tf.keras.Model(inputs, preprocessed_result)
        tf.keras.utils.plot_model(preprocessor,
                                  rankdir="LR",
                                  # show_shapes=True,
                                  to_file='preprocess_layers.png')
        return preprocessor

    def make_tensor_ds_of_preprocessed_data(
            self,
            label_tensor,
            preprocessed_train_features,
            num_examples,
            is_training):
        dataset = tf.data.Dataset.from_tensor_slices((preprocessed_train_features, label_tensor))
        if is_training:
            dataset = dataset.shuffle(num_examples)
            dataset = dataset.repeat()
        dataset = dataset.batch(preprocess_batch_size)
        dataset = dataset.cache().prefetch(buffer_size=preprocess_buffer_size)
        return dataset
