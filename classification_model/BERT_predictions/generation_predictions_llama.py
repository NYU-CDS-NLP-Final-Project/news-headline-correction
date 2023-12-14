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

# Specify hyperparameter version
version = "16batch_5epoch_2e5lr_1wd"

# New headlines 
test = pd.read_csv('/scratch/amh9750/news-headline-correction/headlines/llama2_headlines/test_bodies_llama_headlines.csv')

# Check for GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"device={device}")

# Load the model in from huggingface
finetune_model_name = f'annabellehuether/bert-base-cased-news-{version}'

finetune_model = AutoModelForSequenceClassification.from_pretrained(finetune_model_name)
finetune_model.to(device)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Make the test set a Dataset
generated_dataset = Dataset.from_pandas(test)

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

generation_results = test

# Add predicted labels and probabilities to the test DataFrame
generation_results['predicted_label'] = predicted_labels

for i in range(2):
    generation_results[f'probability_class_{i}'] = [prob[i] for prob in probabilities]

generation_results.to_csv(f'generated_headlines_model_predictions_llama.csv')
