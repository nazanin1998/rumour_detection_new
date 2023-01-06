from abc import ABC


class BertPreprocessing(ABC):

    def start(self, df):
        pass

    def ds_statistics(
            self,
            df,
            label_feature_name,
            categorical_feature_names,
            binary_feature_names,
            numeric_feature_names,
            str_feature_names):
        pass

    def make_input_for_all_ds_columns(
            self,
            df,
            str_feature_names,
            categorical_feature_names,
            binary_feature_names,
            ignore_exc_str):
        pass

    def make_binary_feature_preprocess_layer(
            self,
            preprocess_layers,
            binary_feature_names,
            inputs):
        pass

    def make_numeric_feature_preprocess_layer(
            self,
            preprocess_layers,
            numeric_feature_names,
            numeric_features,
            inputs):
        pass

    def make_string_feature_preprocess_layer(
            self,
            inputs,
            preprocess_layers,
            str_feature_names):
        pass

    def make_preprocess_model(
            self,
            preprocess_layers,
            inputs,
            bert_pack):
        pass

    def make_tensor_ds_of_preprocessed_data(
            self,
            label_tensor,
            preprocessed_train_features,
            num_examples,
            is_training):
        pass


def get_categorical_binary_numeric_string_feature_names():
    categorical_feature_names = []
    str_feature_names = ['text']
    binary_feature_names = ['is_truncated', 'is_source_tweet', 'user.verified', 'user.protected', ]
    numeric_feature_names = ['tweet_id', 'tweet_length', 'symbol_count', 'mentions_count', 'urls_count',
                             'retweet_count', 'favorite_count', 'hashtags_count', 'in_reply_user_id',
                             'in_reply_tweet_id', 'user.id', 'user.name_length', 'user.listed_count',
                             'user.tweets_count', 'user.statuses_count', 'user.friends_count',
                             'user.favourites_count', 'user.followers_count', 'user.follow_request_sent', ]

    label_feature_name = 'is_rumour'
    return label_feature_name, categorical_feature_names, binary_feature_names, numeric_feature_names, str_feature_names
