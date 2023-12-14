import pandas as pd
import numpy as np
from tqdm import tqdm 
tqdm.pandas()

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def summarize_article_t5(text): 

    try: 
        tokens_input = tokenizer.encode("summarize: "+text, return_tensors='pt', max_length=1024, truncation=True)
        summary_ids = model.generate(tokens_input, min_length=80, max_length=100)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        # print(summary)
        return summary

    except Exception as e: 
        print(f"Error processing article: {e}")
        return text

    # return tokens_input

# Pull in train and test data
train_bodies = pd.read_csv('/scratch/mcs9834/llm_env/news-headline-correction/train_bodies_preprocessed_nontokenized.csv')
test_bodies = pd.read_csv('/scratch/mcs9834/llm_env/news-headline-correction/test_bodies_preprocessed_nontokenized.csv')

# Apply PEGASUS summaries to all train and test data
tokenizer = AutoTokenizer.from_pretrained("t5-base") #"csebuetnlp/mT5_multilingual_XLSum")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") #"csebuetnlp/mT5_multilingual_XLSum")

train_bodies_pegasus = train_bodies.copy()
test_bodies_pegasus = test_bodies.copy()

print('summary starting')

train_bodies_pegasus['articleBody'] = train_bodies_pegasus['articleBody'].apply(summarize_article_t5)
test_bodies_pegasus['articleBody'] = test_bodies_pegasus['articleBody'].apply(summarize_article_t5)

print('summary finished')

train_bodies_pegasus.to_csv('/scratch/mcs9834/llm_env/news-headline-correction/train_bodies_summarized_max100tok_t5.csv')
test_bodies_pegasus.to_csv('/scratch/mcs9834/llm_env/news-headline-correction/test_bodies_summarized_max_100tok_t5.csv')

