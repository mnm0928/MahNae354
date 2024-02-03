import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from Bert import preprocess, TextClassificationDataset, BERTClassifier, train, evaluate
from project.preprocessing.text_cleaning import clean_text

"""
Overview:

The BERT model based on the transformer architecture was experimented with to capture the entire text and semantics. 
However, due to time and resource constraints, hyperparameter tuning was not performed. 
The model was run for only 5 epochs, yielding non-acceptable results.

Epoch 1/5
Train Loss:  1.7941284827434034
Validation Loss:  1.7025331263901085
Validation Accuracy: 0.3062

Epoch 2/5
Train Loss:  1.7680091934392101
Validation Loss:  1.7458611253769167
Validation Accuracy: 0.2589

Epoch 3/5
Train Loss:  1.7396467010180154
Validation Loss:  1.7101212398980254
Validation Accuracy: 0.2589

Epoch 4/5
Train Loss:  1.7176653744072043
Validation Loss:  1.6921486775080363
Validation Accuracy: 0.3062

Epoch 5/5
Train Loss:  1.6877402704676419
Validation Loss:  1.664598713382598
Validation Accuracy: 0.3062
---
Testing Accuracy: 0.2972
"""

# read the file
df = pd.read_csv(r"/sample_data.csv")

# Apply text cleaning
df['cleaned_text'] = df['text'].apply(clean_text)

# Apply Preprocessing
texts, labels, label_encoder = preprocess(df)

# Define model parameters
bert_model_name = 'bert-base-uncased'
num_classes = 6
max_length = 8
batch_size = 8
num_epochs = 5
learning_rate = 0.001

# Split the data
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.4, random_state=42)
val_texts, test_texts, val_labels, test_labels = train_test_split(val_texts, val_labels, test_size=0.5, random_state=42)

# Initialize tokenizer, dataset, and data loader
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Set up model
model = BERTClassifier(bert_model_name, num_classes)

# Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training the model
train_losses = []
val_losses = []
test_losses = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = train(model, train_dataloader, optimizer, scheduler)
    train_losses.append(train_loss)
    print("Train Loss: ", train_loss)

    accuracy_val, report_val, val_loss = evaluate(model, val_dataloader)
    print("Validation Loss: ", val_loss)
    print("\n Validation Accuracy: {:.4f}".format(accuracy_val))
    #print(report_val)
    val_losses.append(val_loss)

# Evaluate on the testing set
accuracy_test, report_test, test_loss = evaluate(model, test_dataloader)
print(f"Testing Accuracy: {accuracy_test:.4f}")
print(report_test)
test_losses.append(test_loss)

# Save the final model
torch.save(model.state_dict(), "../saved_models/bert_classifier.pth")

# Plotting the training loss vs validation loss
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss vs Validation Loss')
plt.legend()
plt.show()
