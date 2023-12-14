from datasets import Dataset
import datasets
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import evaluate
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import GroupShuffleSplit
from huggingface_hub import notebook_login
from transformers import TrainingArguments, Trainer
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

seed = 47
torch.manual_seed(seed)
np.random.seed(seed)

# Specify hyperparamter version
version = "16batch_5epoch_2e5lr_1wd"

# New headlines 
generated_headlines = pd.read_csv('/scratch/amh9750/news-headline-correction/bodies_t5_base_headlines.csv')

# Read in data
test_bodies = pd.read_csv('/scratch/amh9750/news-headline-correction/test_bodies_preprocessed_nontokenized.csv')
test_stances = pd.read_csv('/scratch/amh9750/news-headline-correction/test_stances_preprocessed_nontokenized.csv')

# Merge the bodies with the labels and headlines
test = test_stances.merge(test_bodies, how='left', on='Body ID')

# Splitting the test set in half gives test as 25% of the data and val set as 25% of the data
splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=47)
test_inds, val_inds = next(splitter.split(test, groups=test['Body ID']))

val = test.iloc[val_inds]
test = test.iloc[test_inds]

# Drop unnecessary/unnamed and rename columns
val = val.drop(val.columns[0], axis=1)
val = val.drop(val.columns[3], axis=1)
val['headline'] = val['Headline']
val['label'] = val['Stance']
val = val.rename(columns={"Body ID":"body_id", "articleBody":"body"})
val = val.drop(columns=['Headline', 'Stance'])

test = test.drop(test.columns[0], axis=1)
test = test.drop(test.columns[3], axis=1)
test['headline'] = test['Headline']
test['label'] = test['Stance']
test = test.rename(columns={"Body ID":"body_id", "articleBody":"body"})
test = test.drop(columns=['Headline', 'Stance'])

val = val.reset_index(drop=True)
test = test.reset_index(drop=True)

# Get the inference set
unique_bodies = set(test['body_id'].unique())

generated_headlines = generated_headlines[generated_headlines["Body ID"].isin(unique_bodies)]

# Check for GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"device={device}")

# Load the model in from huggingface
finetune_model_name = f'annabellehuether/bert-base-cased-news-{version}'

finetune_model = AutoModelForSequenceClassification.from_pretrained(finetune_model_name)
finetune_model.to(device)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Make the test set a Dataset
generated_dataset = Dataset.from_pandas(generated_headlines)

def tokenize_function(example):
    tokenized_example = tokenizer(
        example["generatedHeadline"],
        example["articleBody"],
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
    )
    return tokenized_example

tokenized_data = generated_dataset.map(
    tokenize_function,
    remove_columns=["Body ID", "generatedHeadline", "articleBody"]
)

tokenized_data.set_format("torch")

finetune_trainer = Trainer(
    model=finetune_model,
    args=TrainingArguments(
        output_dir="./results", 
        per_device_eval_batch_size=16,
        seed = 47
    ),
)

# Get predictions from the model
results = finetune_trainer.predict(tokenized_data)

predicted_labels = torch.argmax(torch.tensor(results.predictions), dim=1).tolist()
probabilities = torch.nn.functional.softmax(torch.tensor(results.predictions), dim=1).tolist()

generation_results = generated_headlines

# Add predicted labels and probabilities to the test DataFrame
generation_results['predicted_label'] = predicted_labels

for i in range(2):
    generation_results[f'probability_class_{i}'] = [prob[i] for prob in probabilities]

generation_results.to_csv(f'generated_headlines_model_predictions_t5.csv')

