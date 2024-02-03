import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import BertModel
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


# Preprocess the dataset
def preprocess(dataframe):
    # Drop rows with missing text and label
    dataframe.dropna(subset=['label'], inplace=True)
    dataframe.dropna(subset=['cleaned_text'], inplace=True)

    # Extract texts and labels
    text_lists = dataframe['cleaned_text'].tolist()
    labels_encoder = LabelEncoder()
    label_lists = labels_encoder.fit_transform(dataframe['label'].tolist())

    return text_lists, label_lists, labels_encoder


# Create a custom dataset class
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length',
                                  truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label)}


# Build BERTClassifier
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logit = self.fc(x)
        return logit


# Define the train() function
def train(model, data_loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return total_loss / len(data_loader)


# Build evaluation method
def evaluate(model, data_loader):
    model.eval()
    predictions = []
    actual_labels = []
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    accuracy = accuracy_score(actual_labels, predictions)
    classification_rep = classification_report(actual_labels, predictions, zero_division=1)
    avg_loss = total_loss / len(data_loader)
    return accuracy, classification_rep, avg_loss
