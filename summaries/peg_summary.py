import pandas as pd
import numpy as np
from tqdm import tqdm 
tqdm.pandas()

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def summarize_article_pegasus(text): 
    try: 
        tokens_input = tokenizer.encode("summarize: "+ text, return_tensors='pt', max_length=512, truncation=True)
        ids = model.generate(tokens_input, min_length=80, max_length=100)
        summary = tokenizer.decode(ids[0], skip_special_tokens=True)
        # print(f'summary: {summary}')
    
        return summary
        # return tokens_input
    
    except Exception as e: 
        print(f'Error processing article {e}')
        return text

# Pull in train and test data
train_bodies = pd.read_csv('/scratch/mcs9834/llm_env/news-headline-correction/train_bodies_preprocessed_nontokenized.csv')
test_bodies = pd.read_csv('/scratch/mcs9834/llm_env/news-headline-correction/test_bodies_preprocessed_nontokenized.csv')

# Apply PEGASUS summaries to all train and test data
model = AutoModelForSeq2SeqLM.from_pretrained('google/pegasus-xsum')
tokenizer = AutoTokenizer.from_pretrained('google/pegasus-xsum') #, use_fast=False)

train_bodies_pegasus = train_bodies.copy()
test_bodies_pegasus = test_bodies.copy()

print('summary starting')

train_bodies_pegasus['articleBody'] = train_bodies_pegasus['articleBody'].apply(summarize_article_pegasus)
test_bodies_pegasus['articleBody'] = test_bodies_pegasus['articleBody'].apply(summarize_article_pegasus)

print('summary completed, writing to csv')

train_bodies_pegasus.to_csv('/scratch/mcs9834/llm_env/news-headline-correction/train_bodies_summarized_max100tok_pegasus.csv')
test_bodies_pegasus.to_csv('/scratch/mcs9834/llm_env/news-headline-correction/test_bodies_summarized_max100tok_pegasus.csv')
