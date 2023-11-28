from datasets import Dataset, load_dataset
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
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from huggingface_hub import notebook_login
from transformers import TrainingArguments, Trainer
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Make sure to login to huggingface on your terminal 

# Read in data
test_bodies = pd.read_csv('/home/amh9750/news-headline-correction/test_bodies_preprocessed_nontokenized.csv')
test_stances = pd.read_csv('/home/amh9750/news-headline-correction/test_stances_preprocessed_nontokenized.csv')
train_bodies = pd.read_csv('/home/amh9750/news-headline-correction/train_bodies_preprocessed_nontokenized.csv')
train_stances = pd.read_csv('/home/amh9750/news-headline-correction/train_stances_preprocessed_nontokenized.csv')

# Merge the bodies with the labels and headlines
train = train_stances.merge(train_bodies, how='left', on='Body ID')
test = test_stances.merge(test_bodies, how='left', on='Body ID')

# Splitting the test set in half gives test as 25% of the data and val set as 25% of the data
splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=47)
test_inds, val_inds = next(splitter.split(test, groups=test['Body ID']))

val = test.iloc[val_inds]
test = test.iloc[test_inds]

# Drop unnecessary/unnamed and rename columns
train = train.drop(train.columns[0], axis=1)
train = train.drop(train.columns[3], axis=1)
train['headline'] = train['Headline']
train['label'] = train['Stance']
train = train.rename(columns={"Body ID":"body_id", "articleBody":"body"})
train = train.drop(columns=['Headline', 'Stance'])

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

train = train.reset_index(drop=True)
val = val.reset_index(drop=True)
test = test.reset_index(drop=True)

# Convert val and train to Datasets
val_dataset = Dataset.from_pandas(val)
train_dataset = Dataset.from_pandas(train)

# Combine into one Dataset dictionary
dd = datasets.DatasetDict({"train":train_dataset,"validation":val_dataset})

# Load in the autotokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(example):
    tokenized_example = tokenizer(
        example["headline"],
        example["body"],
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
    )
    return tokenized_example

# Tokenize the examples
tokenized_datasets = dd.map(
    tokenize_function,
    remove_columns=["body_id", "headline", "body"]
)

# Prepare for training
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Subsets of data
small_train_dataset = tokenized_datasets["train"].shuffle(seed=47).select(range(49972))
small_eval_dataset = tokenized_datasets["validation"].shuffle(seed=47).select(range(12607))
#small_train_dataset = tokenized_datasets["train"].shuffle(seed=47).select(range(100))
#small_eval_dataset = tokenized_datasets["validation"].shuffle(seed=47).select(range(20))

# Load in model
model_name = "bert-base-cased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Check for GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"device={device}")
model.to(device)

# Define training arguments
model_name = f"{model_name}-finetuned-news-all"
training_args = TrainingArguments(output_dir=model_name,
                                  num_train_epochs=3,
                                  learning_rate=2e-5,
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False,
                                  push_to_hub=True,
                                  log_level="error",
                                  seed=47
                                  )

# Define evaluation metric
metric = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Defien the trainer 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

# Fine tune the model
trainer.train()

# Push the model to huggingface
trainer.push_to_hub(commit_message="Training completed!")

# Load the model back in from huggingface
# Replace 'your-username/your-model-name' with your actual username and model name
finetune_model_name = 'annabellehuether/bert-base-cased-finetuned-news-all'

finetune_model = AutoModelForSequenceClassification.from_pretrained(finetune_model_name)
finetune_model.to(device)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Make the test set a Dataset
test_dataset = Dataset.from_pandas(test)

def tokenize_function(example):
    tokenized_example = tokenizer(
        example["headline"],
        example["body"],
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
    )
    return tokenized_example

# Tokenize the test data
tokenized_test = test_dataset.map(
    tokenize_function,
    remove_columns=["body_id", "headline", "body"]
)

# Prepare for training
tokenized_test = tokenized_test.rename_column("label", "labels")
tokenized_test.set_format("torch")

#small_test_dataset = tokenized_test.shuffle(seed=47).select(range(100))
small_test_dataset = tokenized_test

# Set up a trainer object to predict on test set
trainer = Trainer(
    model=finetune_model,
    args=TrainingArguments(
        output_dir="./results",  # Set the directory for saving results
        per_device_eval_batch_size=2,
    ),
)

results = trainer.predict(small_test_dataset)

predictions = np.argmax(results.predictions, axis=1)

true_labels = small_test_dataset["labels"]

precision = precision_score(true_labels, predictions, average='weighted')
recall = recall_score(true_labels, predictions, average='weighted')
f1 = f1_score(true_labels, predictions, average='weighted')
accuracy = accuracy_score(true_labels, predictions)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")

results.metrics

with open('BERT_metrics_full.txt', 'w') as f:
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Accuracy: {accuracy}")

