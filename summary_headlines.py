import torch
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = T5ForConditionalGeneration.from_pretrained("Michau/t5-base-en-generate-headline")
tokenizer = T5Tokenizer.from_pretrained("Michau/t5-base-en-generate-headline")
model = model.to(device)

train_peg_csv_path = '/scratch/mcn8851/news-headline-correction/train_bodies_summarized_pegasus.csv' # Update file path
test_peg_csv_path = '/scratch/mcn8851/news-headline-correction/test_bodies_summarized_pegasus.csv' # Update file path

train_t5_csv_path = '/scratch/mcn8851/news-headline-correction/train_bodies_summarized_t5' # Update file path
test_t5_csv_path = '/scratch/mcn8851/news-headline-correction/test_bodies_summarized_t5' # Update file path

peg_results_csv_path = '/scratch/mcn8851/LLM_env/nlp/summaries_pegasus_headlines.csv' # Update file path
t5_results_csv_path = '/scratch/mcn8851/LLM_env/nlp/summaries_t5_base_headlines.csv' # Update file path

train_peg = pd.read_csv(train_peg_csv_path)
test_peg = pd.read_csv(test_peg_csv_path)

df_peg = pd.concat([train_peg, test_peg], ignore_index=True)

train_t5 = pd.read_csv(train_t5_csv_path)
test_t5 = pd.read_csv(test_t5_csv_path)

df_t5 = pd.concat([train_t5, test_t5], ignore_index=True)

def generate_headline(row):
    article = row['articleBody']
    text = "headline: " + article

    max_len = 256

    encoding = tokenizer.encode_plus(text, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_masks = encoding["attention_mask"].to(device)

    beam_outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        max_length=64,
        num_beams=3,
        early_stopping=True,
    )

    result = tokenizer.decode(beam_outputs[0])
    return result

df_peg['headlines'] = df_peg.apply(generate_headline, axis=1)

df_peg['generatedHeadline'] = df_peg['headlines'].str.replace('<pad>', '').str.replace('</s>', '').str.strip()

df_peg = df_peg[['Body ID', 'articleBody', 'generatedHeadline']]

df_peg.to_csv(peg_results_csv_path, index=False)


df_t5['headlines'] = df_t5.apply(generate_headline, axis=1)

df_t5['generatedHeadline'] = df_t5['headlines'].str.replace('<pad>', '').str.replace('</s>', '').str.strip()

df_t5 = df_t5[['Body ID', 'articleBody', 'generatedHeadline']]

df_t5.to_csv(t5_results_csv_path, index=False)

