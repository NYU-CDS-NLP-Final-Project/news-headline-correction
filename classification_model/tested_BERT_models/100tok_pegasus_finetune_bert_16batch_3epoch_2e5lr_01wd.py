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

# Specify the hyperparameter version
version = "16batch_3epoch_2e5lr_01wd"

# Full test text for validation and testing
test_bodies = pd.read_csv('/scratch/amh9750/news-headline-correction/test_bodies_preprocessed_nontokenized.csv')
test_stances = pd.read_csv('/scratch/amh9750/news-headline-correction/test_stances_preprocessed_nontokenized.csv')

# Summarized train bodies
train_bodies = pd.read_csv('/scratch/amh9750/news-headline-correction/train_bodies_summarized_max100tok_pegasus.csv')
train_stances = pd.read_csv('/scratch/amh9750/news-headline-correction/train_stances_preprocessed_nontokenized.csv')

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

# Tokenize
tokenized_datasets = dd.map(
    tokenize_function,
    remove_columns=["body_id", "headline", "body"]
)

tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Extract correct datasets for training
train_data = tokenized_datasets["train"]
eval_data = tokenized_datasets["validation"]

# Load in model
model_name = "bert-base-cased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Check for GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"device={device}")
model.to(device)

# Define training arguments
model_name = f"100tok-pegasus-summaries-bert-base-cased-news-{version}"
training_args = TrainingArguments(output_dir=model_name,
                                  num_train_epochs=3,
                                  learning_rate=2e-5,
                                  weight_decay=0.01,
                                  per_device_train_batch_size=16,
                                  per_device_eval_batch_size=16,
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False,
                                  push_to_hub=True,
                                  log_level="error",
                                  seed=47
                                  )

# Set up evaluation
metric = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Set up trainer 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    compute_metrics=compute_metrics,
)

# Finetune
trainer.train()

# Push the model to huggingface
trainer.push_to_hub(commit_message="Training completed!")

# Load the model back in from huggingface
finetune_model_name = f'annabellehuether/100tok-pegasus-summaries-bert-base-cased-news-{version}'

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

tokenized_test = tokenized_test.rename_column("label", "labels")
tokenized_test.set_format("torch")

# Set up a trainer object to predict on test set
finetune_trainer = Trainer(
    model=finetune_model,
    args=TrainingArguments(
        output_dir="./results", 
        per_device_eval_batch_size=16,
        seed = 47
    ),
)

# Make predictions on test set
results = finetune_trainer.predict(tokenized_test)

predictions = np.argmax(results.predictions, axis=1)

true_labels = tokenized_test["labels"]

precision = precision_score(true_labels, predictions, average='weighted')
recall = recall_score(true_labels, predictions, average='weighted')
f1 = f1_score(true_labels, predictions, average='weighted')
accuracy = accuracy_score(true_labels, predictions)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")

results.metrics

with open(f'100tok_pegasus_summaries_bert_metrics_{version}.txt', 'w') as f:
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Accuracy: {accuracy}")


predicted_labels = torch.argmax(torch.tensor(results.predictions), dim=1).tolist()
probabilities = torch.nn.functional.softmax(torch.tensor(results.predictions), dim=1).tolist()

test_results = test

# Add predicted labels and probabilities to the test DataFrame
test_results['predicted_label'] = predicted_labels

for i in range(2):
    test_results[f'probability_class_{i}'] = [prob[i] for prob in probabilities]

test_results.to_csv(f'100tok_pegasus_summaries_bert_test_results_{version}.csv')
