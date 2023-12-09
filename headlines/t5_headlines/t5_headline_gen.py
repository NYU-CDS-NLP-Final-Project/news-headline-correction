import torch
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = T5ForConditionalGeneration.from_pretrained("Michau/t5-base-en-generate-headline")
tokenizer = T5Tokenizer.from_pretrained("Michau/t5-base-en-generate-headline")
model = model.to(device)

# Function to generate headlines for each article body
def generate_headline(article_body):
    text = "headline: " + article_body

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

    result = result.replace('<pad>', '').replace('</s>', '').strip()

    return result

# Input paths (Update)
input_path_1 = '/scratch/mcn8851/news-headline-correction/test_bodies_preprocessed_nontokenized.csv'

input_path_2 = '/scratch/mcn8851/news-headline-correction/test_bodies_summarized_pegasus_loop.csv'
input_path_3 = '/scratch/mcn8851/news-headline-correction/test_bodies_summarized_max100tok_pegasus.csv'

input_path_4 = '/scratch/mcn8851/news-headline-correction/test_bodies_summarized_t5'
input_path_5 = '/scratch/mcn8851/news-headline-correction/test_bodies_summarized_max_100tok_t5.csv'

# Output paths (Update)
output_path_1 = '/scratch/mcn8851/LLM_env/nlp/t5_headlines/test_bodies_t5_headlines.csv'

output_path_2 = '/scratch/mcn8851/LLM_env/nlp/t5_headlines/test_bodies_summarized_pegasus_loop_t5_headlines.csv'
output_path_3 = '/scratch/mcn8851/LLM_env/nlp/t5_headlines/test_bodies_summarized_max100tok_pegasus_t5_headlines.csv'

output_path_4 = '/scratch/mcn8851/LLM_env/nlp/t5_headlines/test_bodies_summarized_t5_t5_headlines.csv'
output_path_5 = '/scratch/mcn8851/LLM_env/nlp/t5_headlines/test_bodies_summarized_max_100tok_t5_t5_headlines.csv'

# Reformat input data
def df_reformat(df):
  return df[['Body ID', 'articleBody']]

# Input CSV with articles
df_1 = df_reformat(pd.read_csv(input_path_1))
df_2 = df_reformat(pd.read_csv(input_path_2))
df_3 = df_reformat(pd.read_csv(input_path_3))
df_4 = df_reformat(pd.read_csv(input_path_4))
df_5 = df_reformat(pd.read_csv(input_path_5))

dfs = [df_1, df_2, df_3, df_4, df_5]

# Apply the generate_headline function to each DataFrame
for df in dfs:
    df['generatedHeadline'] = df['articleBody'].apply(generate_headline)

df_1.to_csv(output_path_1, index=False)
df_2.to_csv(output_path_2, index=False)
df_3.to_csv(output_path_3, index=False)
df_4.to_csv(output_path_4, index=False)
df_5.to_csv(output_path_5, index=False)

print(f">> Results dumped to t5 folder!")
