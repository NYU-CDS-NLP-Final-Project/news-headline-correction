import pandas as pd
import sklearn
from sklearn.model_selection import GroupShuffleSplit
from functools import reduce

def test_split(input_path, test_body, new_col_name):
    df = pd.read_csv(input_path)
    df = df[['Body ID', 'generatedHeadline']]
    merged_df = test_body.merge(df, how='inner', on='Body ID')
    merged_df = merged_df[['Body ID', 'articleBody', 'generatedHeadline']].drop_duplicates().reset_index(drop=True)
    merged_df = merged_df.rename(columns={'generatedHeadline': new_col_name})
    return merged_df[['Body ID', new_col_name]]

# Read in data
test_bodies = pd.read_csv('/scratch/mcn8851/news-headline-correction/test_bodies_preprocessed_nontokenized.csv')
test_stances = pd.read_csv('/scratch/mcn8851/news-headline-correction/test_stances_preprocessed_nontokenized.csv')
train_bodies = pd.read_csv('/scratch/mcn8851/news-headline-correction/train_bodies_preprocessed_nontokenized.csv')
train_stances = pd.read_csv('/scratch/mcn8851/news-headline-correction/train_stances_preprocessed_nontokenized.csv')

# Input Paths
input_paths = [
    '/scratch/mcn8851/LLM_env/nlp/llama2_headlines/test_bodies_llama_headlines.csv',
    '/scratch/mcn8851/LLM_env/nlp/llama2_headlines/test_bodies_summarized_200tok_pegasus_llama_headlines.csv',
    '/scratch/mcn8851/LLM_env/nlp/llama2_headlines/test_bodies_summarized_max100tok_pegasus_llama_headlines.csv',
    '/scratch/mcn8851/LLM_env/nlp/llama2_headlines/test_bodies_summarized_t5_llama_headlines.csv',
    '/scratch/mcn8851/LLM_env/nlp/llama2_headlines/test_bodies_summarized_max_100tok_t5_llama_headlines.csv',
    '/scratch/mcn8851/LLM_env/nlp/t5_headlines/test_bodies_t5_headlines.csv',
    '/scratch/mcn8851/LLM_env/nlp/t5_headlines/test_bodies_summarized_200tok_pegasus_t5_headlines.csv',
    '/scratch/mcn8851/LLM_env/nlp/t5_headlines/test_bodies_summarized_max100tok_pegasus_t5_headlines.csv',
    '/scratch/mcn8851/LLM_env/nlp/t5_headlines/test_bodies_summarized_t5_t5_headlines.csv',
    '/scratch/mcn8851/LLM_env/nlp/t5_headlines/test_bodies_summarized_max_100tok_t5_t5_headlines.csv'
]

column_names = [
    'llama_full_headline',
    'llama_pegasus_headline',
    'llama_pegasus100_headline',
    'llama_t5base_headline',
    'llama_t5base100_headline',
    't5_full_headline',
    't5_pegasus_headline',
    't5_pegasus100_headline',
    't5_t5base_headline',
    't5_t5base100_headline'
]


# Merge the bodies with the labels and headlines
train = train_stances.merge(train_bodies, how='left', on='Body ID')
test = test_stances.merge(test_bodies, how='left', on='Body ID')

# Splitting the test set in half gives test as 25% of the data and val set as 25% of the data
splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=47)
test_inds, val_inds = next(splitter.split(test, groups=test['Body ID']))

val = test.iloc[val_inds]
test = test.iloc[test_inds]

test_body = test[['Body ID', 'articleBody']].drop_duplicates().reset_index(drop=True)

# Input CSV with articles
dfs = [test_split(path, test_body, col_name) for path, col_name in zip(input_paths, column_names)]

# Assuming dfs is the list of DataFrames
merged_df = reduce(lambda left, right: pd.merge(left, right, on='Body ID', how='inner'), dfs)
merged_df = merged_df.reset_index(drop=True)
headline_df = test_body.merge(merged_df, how='inner', on='Body ID').reset_index(drop=True)

headline_df.to_csv('/scratch/mcn8851/LLM_env/nlp/eval_headlines/eval_headlines_master.csv', index=False)

print(f">> Results dumped to nlp folder!")
