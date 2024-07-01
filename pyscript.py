import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from transformers import logging

logging.set_verbosity_warning()

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

train_path = "./Data/train.csv"
validation_path = "./Data/validation.csv"
train_df = pd.read_csv(train_path, header=None)
validation_df = pd.read_csv(validation_path, header=None)
train_df.columns = ['abstract', 'labels']
validation_df.columns = ['abstract', 'labels']

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

train_df['abstract'] = train_df['abstract'].apply(preprocess_text)
validation_df['abstract'] = validation_df['abstract'].apply(preprocess_text)

label_mapping = {label: idx for idx, label in enumerate(train_df['labels'].unique())}
train_df['labels'] = train_df['labels'].map(label_mapping)
validation_df['labels'] = validation_df['labels'].map(label_mapping)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(validation_df)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['abstract'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_mapping))

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

eval_results = trainer.evaluate()

print(f"Evaluation results: {eval_results}")

predictions = trainer.predict(val_dataset)
preds = np.argmax(predictions.predictions, axis=1)

accuracy = accuracy_score(validation_df['labels'], preds)
print(f"Accuracy: {accuracy:.2f}")

labels = ['Computation and Language (CL)', 'Cryptography and Security (CR)', 'Distributed and Cluster Computing (DC)', 
          'Data Structures and Algorithms (DS)', 'Logic in Computer Science (LO)', 
          'Networking and Internet Architecture (NI)', 'Software Engineering (SE)']

print(classification_report(validation_df['labels'], preds, target_names=labels))

cm = confusion_matrix(validation_df['labels'], preds)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
