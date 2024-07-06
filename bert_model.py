
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
import torch
from transformers import BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
#change the directory
file_path = '/Users/sehwagvijay/Desktop/BERT/Bert-Based-Job-Description-Classification-and-Skill-Tagging/resume_data/Trimmed_Resume.csv' 
resume_data = pd.read_csv(file_path)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_resumes(resumes):
    return tokenizer.batch_encode_plus(
        resumes,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

tokens = tokenize_resumes(resume_data['Resume_str'].tolist())

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(resume_data['Category'])

labels = torch.tensor(encoded_labels)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'], labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)

optimizer = AdamW(model.parameters(), lr=2e-5)

def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels).item()
    return total_loss / len(dataloader), correct_predictions / len(dataloader.dataset)

epochs = 3
for epoch in range(epochs):
    train_loss = train(model, train_dataloader, optimizer)
    val_loss, val_accuracy = evaluate(model, val_dataloader)
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')


model.save_pretrained('/Users/sehwagvijay/Desktop/BERT/Bert-Based-Job-Description-Classification-and-Skill-Tagging/resume_bert_model')
tokenizer.save_pretrained('/Users/sehwagvijay/Desktop/BERT/Bert-Based-Job-Description-Classification-and-Skill-Tagging/resume_bert_tokenizer')