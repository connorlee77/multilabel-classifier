import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import os
import random
import numpy as np
import tqdm
from sklearn.metrics import accuracy_score

from dataset import MultilabelDataset

### Seeding
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



### Dataset Paths
DATA_PATH = r'C:\Users\conno\Desktop\datasets\miml_dataset'
TRAIN_FILE = r'miml_labels_2.csv'
VALID_FILE = r'miml_labels_val.csv'
IMAGE_FOLDER = r'images'


### Model/Training params
BATCH_SIZE = 64
N_CLASSES = 5

### Transforms
train_transforms = transforms.Compose([
   transforms.RandomHorizontalFlip(),
   transforms.RandomRotation(10),
   transforms.RandomCrop((224, 224), pad_if_needed=True),
   transforms.ToTensor(),
   transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])

test_transforms = transforms.Compose([
   transforms.CenterCrop((224, 224)),
   transforms.ToTensor(),
   transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])

### Load dataset
train_data = MultilabelDataset(DATA_PATH, TRAIN_FILE, IMAGE_FOLDER, N_CLASSES, train_transforms)
valid_data = MultilabelDataset(DATA_PATH, VALID_FILE, IMAGE_FOLDER, N_CLASSES, test_transforms)

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')

train_iterator = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
valid_iterator = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE)


device = torch.device('cuda')

### Build Model
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(in_features=512, out_features=N_CLASSES),
    nn.Sigmoid()
    )

model.to(device)

### Optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()

def calculate_accuracy(yhat, y):
    preds = (yhat > 0.5).float()
    correct = preds.eq(y).sum()
    acc = accuracy_score(y.cpu(), preds.cpu())

    return acc

def train(model, device, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    for (x, y) in tqdm.tqdm(iterator):
        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
                
        yhat = model(x)
        
        loss = criterion(yhat, y)
        acc = calculate_accuracy(yhat, y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, device, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    m = nn.Sigmoid()
    with torch.no_grad():
        for (x, y) in tqdm.tqdm(iterator):

            x = x.to(device)
            y = y.to(device)

            yhat = model(x)

            loss = criterion(yhat, y)
            acc = calculate_accuracy(yhat, y)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


if __name__ == '__main__':
    
    EPOCHS = 10
    SAVE_DIR = 'models'
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'resnet18.pt')

    best_valid_loss = float('inf')

    if not os.path.isdir(f'{SAVE_DIR}'):
        os.makedirs(f'{SAVE_DIR}')

    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, device, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, device, valid_iterator, criterion)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        
        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:05.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:05.2f}% |')
