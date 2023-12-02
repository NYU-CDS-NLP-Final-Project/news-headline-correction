import torch
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = T5ForConditionalGeneration.from_pretrained("Michau/t5-base-en-generate-headline")
tokenizer = T5Tokenizer.from_pretrained("Michau/t5-base-en-generate-headline")
model = model.to(device)

data_csv_path = '/scratch/mcn8851/LLM_env/nlp/bodies_preprocessed_nontokenized.csv' # Update file path
results_csv_path = '/scratch/mcn8851/LLM_env/nlp/bodies_t5_base_headlines.csv' # Update file path

df = pd.read_csv(data_csv_path)

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

df['headlines'] = df.apply(generate_headline, axis=1)

df['generatedHeadline'] = df['headlines'].str.replace('<pad>', '').str.replace('</s>', '').str.strip()

df = df[['Body ID', 'articleBody', 'generatedHeadline']]

df.to_csv(results_csv_path, index=False)

