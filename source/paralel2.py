import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from TorchCRF import CRF
from sklearn.metrics import f1_score
import numpy as np
import os

# Define constants
scenario = 'binary'
output_dir = os.path.join('train_outputs', 'discriminative')
MAX_QUADS = 5
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define aspect categories and sentiment polarities
ASPECT_CATEGORIES = ['component', 'other_category1', 'other_category2']  # Add all your categories
SENTIMENT_POLARITIES = ['positive', 'negative', 'neutral']

# Define IOB2 tags
IOB2_TAGS = ['O', 'B-ASPECT', 'I-ASPECT', 'B-OPINION', 'I-OPINION']
NUM_TAGS = len(IOB2_TAGS)

class QuadrupleDataset(Dataset):
    def __init__(self, filename, tokenizer):
        self.data = []
        self.tokenizer = tokenizer
        
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                text, labels = line.strip().split('####')
                quads = eval(labels)
                self.data.append((text, quads))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, quads = self.data[idx]
        
        # Tokenize input
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=MAX_LENGTH, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Prepare quadruple targets
        iob2_labels = torch.zeros((MAX_QUADS, MAX_LENGTH), dtype=torch.long)
        category_labels = torch.zeros(MAX_QUADS, dtype=torch.long)
        sentiment_labels = torch.zeros(MAX_QUADS, dtype=torch.long)
        quad_mask = torch.zeros(MAX_QUADS, dtype=torch.bool)
        
        for i, (aspect, category, sentiment, opinion) in enumerate(quads[:MAX_QUADS]):
            # Create IOB2 labels
            aspect_tokens = self.tokenizer.encode(aspect, add_special_tokens=False)
            opinion_tokens = self.tokenizer.encode(opinion, add_special_tokens=False)
            
            aspect_start = input_ids.tolist().index(aspect_tokens[0])
            opinion_start = input_ids.tolist().index(opinion_tokens[0])
            
            iob2_labels[i, aspect_start] = IOB2_TAGS.index('B-ASPECT')
            iob2_labels[i, aspect_start+1:aspect_start+len(aspect_tokens)] = IOB2_TAGS.index('I-ASPECT')
            iob2_labels[i, opinion_start] = IOB2_TAGS.index('B-OPINION')
            iob2_labels[i, opinion_start+1:opinion_start+len(opinion_tokens)] = IOB2_TAGS.index('I-OPINION')
            
            category_labels[i] = ASPECT_CATEGORIES.index(category)
            sentiment_labels[i] = SENTIMENT_POLARITIES.index(sentiment)
            quad_mask[i] = True
        
        return input_ids, attention_mask, iob2_labels, category_labels, sentiment_labels, quad_mask

# Prepare datasets and dataloaders
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
if scenario == 'binary':
    train_dataset = QuadrupleDataset(os.path.join('data', 'acos_drone_binary', 'train.txt'), tokenizer)
    test_dataset = QuadrupleDataset(os.path.join('data', 'acos_drone_binary', 'test.txt'), tokenizer)
else:
    train_dataset = QuadrupleDataset(os.path.join('data', 'acos_drone_multi', 'train.txt'), tokenizer)
    test_dataset = QuadrupleDataset(os.path.join('data', 'acos_drone_multi', 'test.txt'), tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

class QuadHead(nn.Module):
    def __init__(self, input_dim, num_categories, num_sentiments):
        super().__init__()
        self.iob2_crf = CRF(NUM_TAGS)
        self.iob2_linear = nn.Linear(input_dim, NUM_TAGS)
        self.category_linear = nn.Linear(input_dim, num_categories)
        self.sentiment_linear = nn.Linear(input_dim, num_sentiments)
    
    def forward(self, x, mask):
        iob2_emissions = self.iob2_linear(x)
        category_logits = self.category_linear(x[:, 0])  # Use [CLS] token for classification
        sentiment_logits = self.sentiment_linear(x[:, 0])  # Use [CLS] token for classification
        
        return iob2_emissions, category_logits, sentiment_logits
    
    def iob2_loss(self, emissions, labels, mask):
        return -self.iob2_crf(emissions, labels, mask=mask)

class QuadrupleExtractionModel(nn.Module):
    def __init__(self, num_categories, num_sentiments):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.quad_heads = nn.ModuleList([QuadHead(self.bert.config.hidden_size, num_categories, num_sentiments) for _ in range(MAX_QUADS)])
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        quad_outputs = [head(last_hidden_state, attention_mask) for head in self.quad_heads]
        return quad_outputs

model = QuadrupleExtractionModel(len(ASPECT_CATEGORIES), len(SENTIMENT_POLARITIES)).to(DEVICE)

def quad_loss(quad_outputs, iob2_labels, category_labels, sentiment_labels, mask):
    total_loss = 0
    for i, (iob2_emissions, category_logits, sentiment_logits) in enumerate(quad_outputs):
        total_loss += model.quad_heads[i].iob2_loss(iob2_emissions, iob2_labels[:, i], mask)
        total_loss += nn.functional.cross_entropy(category_logits, category_labels[:, i], ignore_index=-1)
        total_loss += nn.functional.cross_entropy(sentiment_logits, sentiment_labels[:, i], ignore_index=-1)
    return total_loss

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

def train(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, iob2_labels, category_labels, sentiment_labels, quad_mask = [b.to(DEVICE) for b in batch]
        
        optimizer.zero_grad()
        quad_outputs = model(input_ids, attention_mask)
        loss = quad_loss(quad_outputs, iob2_labels, category_labels, sentiment_labels, attention_mask)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}")

def evaluate(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, iob2_labels, category_labels, sentiment_labels, quad_mask = [b.to(DEVICE) for b in batch]
            
            quad_outputs = model(input_ids, attention_mask)
            
            for i, (iob2_emissions, category_logits, sentiment_logits) in enumerate(quad_outputs):
                iob2_preds = model.quad_heads[i].iob2_crf.decode(iob2_emissions, attention_mask)
                category_preds = category_logits.argmax(dim=1)
                sentiment_preds = sentiment_logits.argmax(dim=1)
                
                all_preds.extend(zip(iob2_preds, category_preds.cpu().numpy(), sentiment_preds.cpu().numpy()))
                all_labels.extend(zip(iob2_labels[:, i].cpu().numpy(), category_labels[:, i].cpu().numpy(), sentiment_labels[:, i].cpu().numpy()))
    
    # Calculate F1 score (you might want to implement a more specific metric for your task)
    f1 = f1_score(all_labels, all_preds, average='micro')
    print(f"F1 Score: {f1:.4f}")
    return f1

# Training loop
best_f1 = 0
for epoch in range(EPOCHS):
    train(model, train_loader, optimizer, epoch)
    f1 = evaluate(model, test_loader)
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))

print(f"Best F1 Score: {best_f1:.4f}")

def inference(model, tokenizer, text):
    model.eval()
    encoding = tokenizer(text, padding='max_length', truncation=True, max_length=MAX_LENGTH, return_tensors='pt')
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    
    with torch.no_grad():
        quad_outputs = model(input_ids, attention_mask)
    
    quadruples = []
    for iob2_emissions, category_logits, sentiment_logits in quad_outputs:
        iob2_pred = model.quad_heads[0].iob2_crf.decode(iob2_emissions, attention_mask)[0]
        category_pred = category_logits.argmax(dim=1).item()
        sentiment_pred = sentiment_logits.argmax(dim=1).item()
        
        aspect_tokens = []
        opinion_tokens = []
        current_tag = 'O'
        for i, tag_idx in enumerate(iob2_pred):
            tag = IOB2_TAGS[tag_idx]
            if tag.startswith('B-'):
                if current_tag != 'O':
                    if current_tag == 'ASPECT':
                        aspect_tokens.append(tokenizer.decode([input_ids[0, i-1]]))
                    else:
                        opinion_tokens.append(tokenizer.decode([input_ids[0, i-1]]))
                current_tag = tag[2:]
            elif tag.startswith('I-'):
                if current_tag == tag[2:]:
                    if current_tag == 'ASPECT':
                        aspect_tokens.append(tokenizer.decode([input_ids[0, i]]))
                    else:
                        opinion_tokens.append(tokenizer.decode([input_ids[0, i]]))
            elif tag == 'O' and current_tag != 'O':
                if current_tag == 'ASPECT':
                    aspect_tokens.append(tokenizer.decode([input_ids[0, i-1]]))
                else:
                    opinion_tokens.append(tokenizer.decode([input_ids[0, i-1]]))
                current_tag = 'O'
        
        if aspect_tokens and opinion_tokens:
            aspect = ' '.join(aspect_tokens)
            opinion = ' '.join(opinion_tokens)
            category = ASPECT_CATEGORIES[category_pred]
            sentiment = SENTIMENT_POLARITIES[sentiment_pred]
            quadruples.append((aspect, category, sentiment, opinion))
    
    return quadruples

# Example usage
model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
text = "Barometer Dead in Air."
result = inference(model, tokenizer, text)
print(result)