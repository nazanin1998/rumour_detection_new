import pandas as pd
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('../../../dataset/yelp_review_polarity_csv/train.csv', header=None)

print(train_df.head())

test_df = pd.read_csv('../../../dataset/yelp_review_polarity_csv/test.csv', header=None)

print(test_df.head())

train_df[0] = (train_df[0] == 2).astype(int)
test_df[0] = (test_df[0] == 2).astype(int)

print(train_df.head())
print(test_df.head())

# Creating training dataframe according to BERT by adding the required columns
df_bert = pd.DataFrame({
    'id': range(len(train_df)),
    'label': train_df[0],
    'alpha': ['a'] * train_df.shape[0],
    'text': train_df[1].replace(r'\n', ' ', regex=True)
})

# Splitting training data file into *train* and *dev*
df_bert_train, df_bert_dev = train_test_split(df_bert, test_size=0.01)
print(df_bert_train.head())

# Creating test dataframe according to BERT
df_bert_test = pd.DataFrame({
    'id': range(len(test_df)),
    'text': test_df[1].replace(r'\n', ' ', regex=True)
})

print(df_bert_test.head())
# Saving dataframes to .tsv format as required by BERT
df_bert_train.to_csv('../../../dataset/yelp_review_polarity_csv/data/train.tsv', sep='\t', index=False, header=False)
df_bert_dev.to_csv('../../../dataset/yelp_review_polarity_csv/data/dev.tsv', sep='\t', index=False, header=False)
df_bert_test.to_csv('../../../dataset/yelp_review_polarity_csv/data/test.tsv', sep='\t', index=False, header=False)
