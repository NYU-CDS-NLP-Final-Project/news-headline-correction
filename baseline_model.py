import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize
import sklearn
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupShuffleSplit
import gensim
import gensim.downloader
from gensim.models import Word2Vec
import tqdm
from tqdm.notebook import tqdm
import sys
import pickle

SEED = 7

# Specify the file path where you want to save the logs
log_file_path = '/home/amr10211/news-headline-correction/log_file_baseline_model.txt'

# Redirect stdout and stderr to the log file
sys.stdout = open(log_file_path, 'w')
sys.stderr = open(log_file_path, 'w')

#Function to tokenize and vectorize
def tokenize_vectorize_mean(data, model):
    """
    Tokenizes and converts the words in each document into their Word2Vec embeddings
    Takes the mean across word embeddings to give a 2D array rather than a 3D array
    param: data (pandas Series)
    param: model (Word2Vec model)
    return: word_vectors (list of arrays)
    """
    # Tokenize the documents
    tokenized_data =[word_tokenize(doc.lower()) for doc in data]

    # Convert tokens to word vectors and take the mean across words to get document vector representations
    if model == pretrained_w2v_model:
        word_vectors = [np.mean([model[token] for token in doc if token in model], axis=0) for doc in tokenized_data]
    else:
        word_vectors = [np.mean([model.wv[token] for token in doc if token in model.wv], axis=0) for doc in tokenized_data]

    return word_vectors

# Load Word2Vec models
pretrained_w2v_model = gensim.downloader.load('word2vec-google-news-300')
datatrained_w2v_model =  Word2Vec.load("w2v_model.bin")
#Set Word2Vec model
w2v_model = pretrained_w2v_model

#Load Data
test_bodies = pd.read_csv('test_bodies_preprocessed_nontokenized.csv')
test_stances = pd.read_csv('test_stances_preprocessed_nontokenized.csv')
train_bodies = pd.read_csv('train_bodies_preprocessed_nontokenized.csv')
train_stances = pd.read_csv('train_stances_preprocessed_nontokenized.csv')

# Merge the bodies with the labels and headlines
train = train_stances.merge(train_bodies, how='left', on='Body ID')
test = test_stances.merge(test_bodies, how='left', on='Body ID')

# Splitting the test set in half gives test as 25% of the data and val set as 25% of the data
splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=47)
test_inds, val_inds = next(splitter.split(test, groups=test['Body ID']))

val = test.iloc[val_inds]
test = test.iloc[test_inds]

# Drop unnecessary/unnamed and rename columns
train = train.drop(train.columns[0], axis=1)
train = train.drop(train.columns[3], axis=1)
train['headline'] = train['Headline']
train['label'] = train['Stance']
train = train.rename(columns={"Body ID":"body_id", "articleBody":"body"})
train = train.drop(columns=['Headline', 'Stance'])

val = val.drop(val.columns[0], axis=1)
val = val.drop(val.columns[3], axis=1)
val['headline'] = val['Headline']
val['label'] = val['Stance']
val = val.rename(columns={"Body ID":"body_id", "articleBody":"body"})
val = val.drop(columns=['Headline', 'Stance'])
val = val[val['headline']!='Crabzilla'] #Remove headline causing error

test = test.drop(test.columns[0], axis=1)
test = test.drop(test.columns[3], axis=1)
test['headline'] = test['Headline']
test['label'] = test['Stance']
test = test.rename(columns={"Body ID":"body_id", "articleBody":"body"})
test = test.drop(columns=['Headline', 'Stance'])

train = train.reset_index(drop=True)
val = val.reset_index(drop=True)
test = test.reset_index(drop=True)

print(train.shape)
print(val.shape)
print(test.shape)

# Extract X vals
X_train_headline = train['headline']
X_val_headline = val['headline']
X_test_headline = test['headline']
X_train_body = train['body']
X_val_body = val['body']
X_test_body = test['body']

# Extract labels
y_train = train['label']
y_val = val['label']
y_test = test['label']

# Concatenate train and val data for cross validation in GridSearchCV
X_headline = pd.concat([X_train_headline, X_val_headline], axis=0)
X_body = pd.concat([X_train_body, X_val_body], axis=0)
y = pd.concat([y_train, y_val], axis=0)

# Vectorize documents
X_headline = pd.Series(tokenize_vectorize_mean(X_headline , w2v_model))
X_body = pd.Series(tokenize_vectorize_mean(X_body , w2v_model))
X_headline = np.vstack(X_headline)
X_body = np.vstack(X_body)
X_combined = np.hstack([X_headline, X_body])

# Change labels to 0 and 1 
y = np.array(y)
y = (y == 1.0).astype(int)

# Initialize XGBoost model
model = xgb.XGBClassifier(random_state=7,  tree_method='hist', device='gpu')

# Define parameter grid to search over
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of boosting stages (trees)
    'max_depth': [5, 10, 20]  # Maximum depth of individual trees
}

print('starting grid search...')
# Initialize and run GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1', verbose=10, error_score='raise')
grid_search.fit(X_combined, y)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
best_f1 = grid_search.best_score_

#Save best model
best_model_filename = "baseline_model.pkl"
with open(best_model_filename, 'wb') as file:
    pickle.dump(best_model, file)
    
# Write best hyperparameters out to a text file
output_file = "baseline_model.txt"

with open(output_file, 'w') as file:
    file.write("Best Hyperparameters:\n")
    for param, value in best_params.items():
        file.write(f"{param}: {value}\n")
    file.write(f"\nBest F1 Score: {best_f1:.4f}\n")

print(f"Best hyperparameters and accuracy written to {output_file}")

