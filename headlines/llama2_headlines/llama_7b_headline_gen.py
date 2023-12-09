from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import peft
from peft import PeftModel
import torch
import datasets
from datasets import load_dataset
import pandas as pd
import re
import os

# First install requirements_llama.txt file in singularity (activate environment)

# Before running SBATCH, set auth_token in command line: export MY_AUTH_TOKEN="hf_..."

# Access the environment variable
hf_token = os.environ.get('MY_AUTH_TOKEN')

# Check if the auth token is available
if hf_token is None:
    raise ValueError("Auth token is not set. Please set the MY_AUTH_TOKEN environment variable.")

model_name = 'TinyPixel/Llama-2-7B-bf16-sharded'
push_to_hub = True
repo_id = "marynwangwu/llama-2-7b-tinypixel" # Make sure repo is public and accessible

os.environ["MODEL_NAME"] = model_name
os.environ["PUSH_TO_HUB"] = str(push_to_hub)
os.environ["HF_TOKEN"] = hf_token
os.environ["REPO_ID"] = repo_id

# 1. Loading model using Peft - from_pretrained() method
# load tokenizer files from checkpoint
tokenizer = AutoTokenizer.from_pretrained(repo_id, token=hf_token)

# load base model used for fine-tuning
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = torch.float16, device_map="auto") #Base_Model for example: meta-llama/Llama-2-13b-chat-hf

# load adapter weights, base model weights using function from_pretrained()
model = PeftModel.from_pretrained(model, repo_id, device_map="auto")

# we use the same formatted text used in fine-tuning to create prompt template
format_text = "You summarize below news article into concise, descriptive news headline that captures the details of the article.\n### article: {}\n### headline: {}"

# Function to generate headlines for each article body
def generate_headline(article_body):
    formatted_input = format_text.format(article_body, "")
    
    # Tokenize the input text
    encoding = tokenizer(formatted_input, return_tensors='pt', truncation=True, max_length=1024)
    # Extract input_ids from the encoding dictionary
    input_ids = encoding['input_ids']
    
    # Calculate the maximum length based on the input and an estimated number of words for the headline
#    estimated_headline_length = 50  # Adjust this value based on your expectation of the headline length
#    input_length = input_ids.size(1)  # Adjust this value based on your maximum input length requirement
#    max_output_length = input_length + estimated_headline_length
    
    # Perform inference
    inputs = tokenizer(formatted_input, return_tensors="pt").to("cuda:0")
    outputs = model.generate(**inputs, max_new_tokens=64, num_beams=3)

    # Define a regular expression pattern
    pattern = r'headline: (.+?)<s>'
    
    # Use re.search to find the match
    match = re.search(pattern, tokenizer.decode(outputs[0]))
    
    # Extract the captured group if a match is found
    if match:
        return match.group(1)
    else:
        return "No match found."

# Input paths (Update)
input_path_1 = '/scratch/mcn8851/news-headline-correction/test_bodies_preprocessed_nontokenized.csv'

input_path_2 = '/scratch/mcn8851/news-headline-correction/test_bodies_summarized_pegasus_loop.csv'
input_path_3 = '/scratch/mcn8851/news-headline-correction/test_bodies_summarized_max100tok_pegasus.csv'

input_path_4 = '/scratch/mcn8851/news-headline-correction/test_bodies_summarized_t5'
input_path_5 = '/scratch/mcn8851/news-headline-correction/test_bodies_summarized_max_100tok_t5.csv'

# Output paths (Update)
output_path_1 = '/scratch/mcn8851/LLM_env/nlp/llama2_headlines/test_bodies_llama_headlines.csv'

output_path_2 = '/scratch/mcn8851/LLM_env/nlp/llama2_headlines/test_bodies_summarized_pegasus_loop_llama_headlines.csv'
output_path_3 = '/scratch/mcn8851/LLM_env/nlp/llama2_headlines/test_bodies_summarized_max100tok_pegasus_llama_headlines.csv'

output_path_4 = '/scratch/mcn8851/LLM_env/nlp/llama2_headlines/test_bodies_summarized_t5_llama_headlines.csv'
output_path_5 = '/scratch/mcn8851/LLM_env/nlp/llama2_headlines/test_bodies_summarized_max_100tok_t5_llama_headlines.csv'

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

print(f">> Results dumped to llama2_headlines folder!")
