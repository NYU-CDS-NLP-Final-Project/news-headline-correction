import pandas as pd
import numpy as np
import nltk
from tqdm import tqdm 
from nltk import word_tokenize
tqdm.pandas()
nltk.download('punkt')

#Function to tokenize data
def tokenize_data(text):
    return word_tokenize(text)

#Pull in Data (update file paths)
train_bodies = pd.read_csv('/Users/allisonredfern/Documents/news-headline-correction/fnc-1-master/train_bodies.csv')
train_stances = pd.read_csv('/Users/allisonredfern/Documents/news-headline-correction/fnc-1-master/train_stances.csv')
test_bodies = pd.read_csv('/Users/allisonredfern/Documents/news-headline-correction/fnc-1-master/competition_test_bodies.csv')
test_stances = pd.read_csv('/Users/allisonredfern/Documents/news-headline-correction/fnc-1-master/competition_test_stances.csv')

#Aggregate Labels (agree, disagree, unrealted, discuss)
mapping = {'agree': 1, 'discuss': 1, 'disagree': 0, 'unrelated': 0}
train_stances['Stance'] = train_stances['Stance'].replace(mapping)
test_stances['Stance'] = test_stances['Stance'].replace(mapping)

#Tokenize Text 
train_bodies['articleBody'] = train_bodies['articleBody'].progress_apply(tokenize_data)
test_bodies['articleBody'] = test_bodies['articleBody'].progress_apply(tokenize_data)
train_stances['Headline'] = train_stances['Headline'].progress_apply(tokenize_data)
test_stances['Headline'] = test_stances['Headline'].progress_apply(tokenize_data)


#Write to csv (update file paths)
train_bodies.to_csv('/Users/allisonredfern/Documents/news-headline-correction/train_bodies_preprocessed.csv')
train_stances.to_csv('/Users/allisonredfern/Documents/news-headline-correction/train_stances_preprocessed.csv')
test_bodies.to_csv('/Users/allisonredfern/Documents/news-headline-correction/test_bodies_preprocessed.csv')
test_stances.to_csv('/Users/allisonredfern/Documents/news-headline-correction/test_stances_preprocessed.csv')
