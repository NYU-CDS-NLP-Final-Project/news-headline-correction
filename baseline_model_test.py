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
from sklearn.metrics import f1_score


SEED = 7

# Specify the file path where you want to save the logs
log_file_path = '/home/amr10211/news-headline-correction/log_file_baseline_model_test.txt'

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

# Merge the bodies with the labels and headlines
test = test_stances.merge(test_bodies, how='left', on='Body ID')

# Splitting the test set in half gives test as 25% of the data and val set as 25% of the data
splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=47)
test_inds, val_inds = next(splitter.split(test, groups=test['Body ID']))

val = test.iloc[val_inds]
test = test.iloc[test_inds]

# Drop unnecessary/unnamed and rename columns
test = test.drop(test.columns[0], axis=1)
test = test.drop(test.columns[3], axis=1)
test['headline'] = test['Headline']
test['label'] = test['Stance']
test = test.rename(columns={"Body ID":"body_id", "articleBody":"body"})
test = test.drop(columns=['Headline', 'Stance'])
test = test[test['headline']!='Crabzilla'] #Remove headline causing error

test = test.reset_index(drop=True)


# Extract X vals
X_test_headline = test['headline']
X_test_body = test['body']

# Extract labels
y_test = test['label']

# Vectorize documents
X_headline = pd.Series(tokenize_vectorize_mean(X_test_headline , w2v_model))
X_body = pd.Series(tokenize_vectorize_mean(X_test_body, w2v_model))
X_headline = np.vstack(X_headline)
X_body = np.vstack(X_body)
X_combined = np.hstack([X_headline, X_body])

# Change labels to 0 and 1 
y = np.array(y_test)
y = (y == 1.0).astype(int)                           

# Load the saved model using pickle
best_model_filename = "baseline_model.pkl"
with open(best_model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Predict on the test set
y_test_pred = loaded_model.predict(X_combined)


# Calculate and print the test F1 score
test_f1_score = f1_score(y, y_test_pred)
print(f"Test F1 Score: {test_f1_score:.4f}")

output_file = "baseline_model_test_results.txt"

# Write test F1score to the output file
with open(output_file, 'a') as file:
    file.write(f"\nTest F1 Score: {test_f1_score:.4f}\n")                                     
