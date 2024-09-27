import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from torchcrf import CRF
from sklearn.preprocessing import MultiLabelBinarizer
from seqeval.metrics import classification_report
import numpy as np
from tqdm import tqdm

# Dataset preparation
class QuadrupleABSADataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlb = MultiLabelBinarizer()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                text, quads = line.strip().split('####')
                self.data.append((text, eval(quads)))
        
        all_aspect_sentiments = [(q[1], q[2]) for item in self.data for q in item[1]]
        self.mlb.fit(all_aspect_sentiments)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, quads = self.data[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        
        # Prepare aspect and opinion labels
        aspect_labels = ['O'] * self.max_len
        opinion_labels = ['O'] * self.max_len
        
        for quad in quads:
            aspect_term, _, _, opinion_term = quad
            
            # Tokenize aspect and opinion terms
            aspect_tokens = self.tokenizer.tokenize(aspect_term)
            opinion_tokens = self.tokenizer.tokenize(opinion_term)
            
            # Find aspect term in tokenized text and label
            for i in range(len(encoding['input_ids'][0]) - len(aspect_tokens)):
                if encoding['input_ids'][0][i:i+len(aspect_tokens)].tolist() == self.tokenizer.convert_tokens_to_ids(aspect_tokens):
                    aspect_labels[i] = 'B-ASP'
                    for j in range(1, len(aspect_tokens)):
                        aspect_labels[i+j] = 'I-ASP'
                    break
            
            # Find opinion term in tokenized text and label
            for i in range(len(encoding['input_ids'][0]) - len(opinion_tokens)):
                if encoding['input_ids'][0][i:i+len(opinion_tokens)].tolist() == self.tokenizer.convert_tokens_to_ids(opinion_tokens):
                    opinion_labels[i] = 'B-OPN'
                    for j in range(1, len(opinion_tokens)):
                        opinion_labels[i+j] = 'I-OPN'
                    break
        
        # Convert labels to ids
        aspect_label_ids = [['O', 'B-ASP', 'I-ASP'].index(label) for label in aspect_labels]
        opinion_label_ids = [['O', 'B-OPN', 'I-OPN'].index(label) for label in opinion_labels]
        
        # Prepare joint aspect-sentiment labels
        aspect_sentiments = [(q[1], q[2]) for q in quads]
        joint_labels = self.mlb.transform([aspect_sentiments])[0]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'aspect_labels': torch.tensor(aspect_label_ids),
            'opinion_labels': torch.tensor(opinion_label_ids),
            'joint_labels': torch.tensor(joint_labels, dtype=torch.float)
        }

# Model definition
class QuadrupleABSAModel(nn.Module):
    def __init__(self, num_labels):
        super(QuadrupleABSAModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.aspect_classifier = nn.Linear(self.bert.config.hidden_size, 3)
        self.opinion_classifier = nn.Linear(self.bert.config.hidden_size, 3)
        self.joint_classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.aspect_crf = CRF(3, batch_first=True)
        self.opinion_crf = CRF(3, batch_first=True)

    def forward(self, input_ids, attention_mask, aspect_labels=None, opinion_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        aspect_emissions = self.aspect_classifier(sequence_output)
        opinion_emissions = self.opinion_classifier(sequence_output)
        joint_logits = self.joint_classifier(pooled_output)

        outputs = {'joint_logits': joint_logits}

        if aspect_labels is not None and opinion_labels is not None:
            aspect_loss = -self.aspect_crf(aspect_emissions, aspect_labels, mask=attention_mask.bool())
            opinion_loss = -self.opinion_crf(opinion_emissions, opinion_labels, mask=attention_mask.bool())
            outputs['aspect_loss'] = aspect_loss
            outputs['opinion_loss'] = opinion_loss

        return outputs

    def decode(self, input_ids, attention_mask):
        outputs = self.forward(input_ids, attention_mask)
        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        aspect_emissions = self.aspect_classifier(sequence_output)
        opinion_emissions = self.opinion_classifier(sequence_output)
        
        aspect_tags = self.aspect_crf.decode(aspect_emissions, mask=attention_mask.bool())
        opinion_tags = self.opinion_crf.decode(opinion_emissions, mask=attention_mask.bool())
        joint_preds = torch.sigmoid(outputs['joint_logits']) > 0.5
        return aspect_tags, opinion_tags, joint_preds

# Loss definition
class QuadrupleABSALoss(nn.Module):
    def __init__(self, aspect_weight=1.0, opinion_weight=1.0, classification_weight=1.0):
        super(QuadrupleABSALoss, self).__init__()
        self.aspect_weight = aspect_weight
        self.opinion_weight = opinion_weight
        self.classification_weight = classification_weight
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, aspect_labels, opinion_labels, joint_labels):
        aspect_loss = outputs['aspect_loss']
        opinion_loss = outputs['opinion_loss']
        classification_loss = self.bce_loss(outputs['joint_logits'], joint_labels)
        
        total_loss = (
            self.aspect_weight * aspect_loss +
            self.opinion_weight * opinion_loss +
            self.classification_weight * classification_loss
        )
        return total_loss

# Training function
def train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, num_epochs):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            aspect_labels = batch['aspect_labels'].to(device)
            opinion_labels = batch['opinion_labels'].to(device)
            joint_labels = batch['joint_labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, aspect_labels, opinion_labels)
            loss = loss_fn(outputs, aspect_labels, opinion_labels, joint_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Average train loss: {avg_train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model")

# Evaluation function
def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            aspect_labels = batch['aspect_labels'].to(device)
            opinion_labels = batch['opinion_labels'].to(device)
            joint_labels = batch['joint_labels'].to(device)

            outputs = model(input_ids, attention_mask, aspect_labels, opinion_labels)
            loss = loss_fn(outputs, aspect_labels, opinion_labels, joint_labels)
            total_loss += loss.item()

    return total_loss / len(data_loader)

# Inference function
def inference(model, tokenizer, text, device, mlb):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        aspect_tags, opinion_tags, joint_preds = model.decode(input_ids, attention_mask)

    # Convert tags to terms
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    aspect_terms = extract_terms(tokens, aspect_tags[0])
    opinion_terms = extract_terms(tokens, opinion_tags[0])

    # Convert joint predictions to aspect-sentiment pairs
    aspect_sentiments = mlb.inverse_transform(joint_preds.cpu().numpy())[0]

    # Combine results
    results = []
    for aspect_term in aspect_terms:
        for opinion_term in opinion_terms:
            for aspect_sentiment in aspect_sentiments:
                results.append([aspect_term, aspect_sentiment[0], aspect_sentiment[1], opinion_term])

    return results

def extract_terms(tokens, tags):
    terms = []
    current_term = []
    for token, tag in zip(tokens, tags):
        if tag == 1:  # B-tag
            if current_term:
                terms.append(' '.join(current_term))
                current_term = []
            current_term.append(token)
        elif tag == 2:  # I-tag
            current_term.append(token)
        elif tag == 0 and current_term:  # O-tag
            terms.append(' '.join(current_term))
            current_term = []
    if current_term:
        terms.append(' '.join(current_term))
    return terms

# Main execution
def main():
    # Hyperparameters
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Prepare datasets
    train_path = os.path.join('data', 'acos_drone_binary', 'train.txt')
    test_path = os.path.join('data', 'acos_drone_binary', 'test.txt')
    train_dataset = QuadrupleABSADataset(train_path, tokenizer, MAX_LEN)
    val_dataset = QuadrupleABSADataset(test_path, tokenizer, MAX_LEN)
    test_dataset = QuadrupleABSADataset(test_path, tokenizer, MAX_LEN)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize model
    num_labels = len(train_dataset.mlb.classes_)
    model = QuadrupleABSAModel(num_labels).to(DEVICE)

    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Initialize loss function
    loss_fn = QuadrupleABSALoss()

    # Train the model
    train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, DEVICE, EPOCHS)

    # Evaluate on test set
    test_loss = evaluate(model, test_loader, loss_fn, DEVICE)
    print(f"Test loss: {test_loss:.4f}")

    # Example inference
    text = "The pizza was delicious but the service was slow."
    results = inference(model, tokenizer, text, DEVICE, train_dataset.mlb)
    print("Inference results:")
    for result in results:
        print(result)

if __name__ == "__main__":
    main()