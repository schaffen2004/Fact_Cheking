import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

class SentencePairDataset(Dataset):
    def __init__(self, sentence_pairs, labels, tokenizer, max_length):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, idx):
        sentence1, sentence2, len_e = self.sentence_pairs[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            sentence1,
            text_pair=sentence2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,  # Tránh lỗi truncation
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),  # Loại bỏ chiều batch
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

class PhoBERTClassifier(nn.Module):
    def __init__(self, phobert, num_classes):
        super(PhoBERTClassifier, self).__init__()
        self.phobert = phobert
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(self.phobert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        dropout_output = self.dropout(pooled_output)
        logits = self.linear(dropout_output)
        return logits

def load_data(type):
    import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
    splits = {'train': 'train.csv', 'validation': 'dev.csv', 'test': 'test.csv'}
    df = pd.read_csv("hf://datasets/schaffen49/ViFactCheck_Combine/" + splits["train"])
    X1 = df['Statement']
    X2 = df['Context']
    X3 = df['len_evidence']
    X = [(x1, x2, x3) for x1, x2, x3 in zip(X1, X2, X3)]
    y = list(df['labels'])
    return X, y

def prepare_datasets(X_train,y_train,X_test,y_test,X_dev,y_dev, tokenizer, max_length):

    train_dataset = SentencePairDataset(X_train, y_train, tokenizer, max_length)
    dev_dataset = SentencePairDataset(X_dev, y_dev, tokenizer, max_length)
    test_dataset = SentencePairDataset(X_test, y_test, tokenizer, max_length)

    return train_dataset, dev_dataset, test_dataset

def train(model, train_loader, dev_loader, criterion, optimizer, device, epochs):
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        predictions = []
        true_labels = []
        for batch in tqdm(dev_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy().tolist())
                true_labels.extend(labels.cpu().numpy().tolist())

        print(f"Epoch {epoch+1}/{epochs}")
        print(classification_report(true_labels, predictions, digits=4))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    max_length = 256
    batch_size = 16
    # epochs = 10
    epochs = 1
    
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large",verbose=False)
    phobert = AutoModel.from_pretrained("vinai/phobert-large")
    
    X_train,y_train = load_data("train")
    X_test,y_test = load_data("test")
    X_dev,y_dev = load_data("dev")
    train_dataset, dev_dataset, test_dataset = prepare_datasets(X_train,y_train,X_test,y_test,X_dev,y_dev, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = PhoBERTClassifier(phobert, num_classes=len(set(y_train))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.5e-5)

    train(model, train_loader, dev_loader, criterion, optimizer, device, epochs)