import os
import glob
import numpy as np
import pandas as pd
import PIL.Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Constants and Configuration
# ---------------------------------------------------------
SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)

MAIN_DIR = './data/'
TRAIN_IMG_DIR = './data/train'
CLASSES_NUM = 6
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
N_TILES = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 2. Data Loading and Preprocessing
# ---------------------------------------------------------
def prepare_data():
    train_csv = pd.read_csv(os.path.join(MAIN_DIR, 'train.csv'))
    
    # Filter valid images (ensuring tile 0 exists)
    valid_images = glob.glob(os.path.join(TRAIN_IMG_DIR, '*_0.png'))
    valid_ids = {os.path.basename(x).replace('_0.png', '') for x in valid_images}
    train_csv = train_csv[train_csv['image_id'].isin(valid_ids)]
    print(f"Total valid training samples: {len(train_csv)}")
    print(len(valid_ids))
    # Split by provider for balanced sets
    radboud_csv = train_csv[train_csv['data_provider'] == 'radboud']
    karolinska_csv = train_csv[train_csv['data_provider'] != 'radboud']
    
    r_train, r_val = train_test_split(radboud_csv, test_size=0.2, random_state=SEED)
    k_train, k_val = train_test_split(karolinska_csv, test_size=0.2, random_state=SEED)
    print(f"Radboud train: {len(r_train)}, val: {len(r_val)}")
    return pd.concat([r_train, k_train]), pd.concat([r_val, k_val])

class PandaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.image_ids = df['image_id'].values
        self.labels = df['isup_grade'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        label = self.labels[idx]
        
        # Load and stack tiles
        tiles = []
        for i in range(N_TILES):
            fname = f"{img_id}_{i}.png"
            img = PIL.Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
            tiles.append(np.array(img))
        
        # Stack vertically to match (1536, 128, 3)
        full_img = np.vstack(tiles) 
        
        if self.transform:
            full_img = self.transform(full_img)
        else:
            full_img = transforms.ToTensor()(full_img)
            
        return full_img, torch.tensor(label, dtype=torch.long)

# ---------------------------------------------------------
# 3. Model Definition
# ---------------------------------------------------------
class ProstateResNet(nn.Module):
    def __init__(self, num_classes=6):
        super(ProstateResNet, self).__init__()
        # Using pre-trained ResNet50 as in the notebook
        self.base_model = models.resnet50(pretrained=True)
        # Replace the final fully connected layer
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# ---------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------
def train_model():
    train_df, valid_df = prepare_data()
    
    # Augmentation transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = PandaDataset(train_df, TRAIN_IMG_DIR, transform=train_transform)
    val_ds = PandaDataset(valid_df[:100], TRAIN_IMG_DIR, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = ProstateResNet(CLASSES_NUM).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        accuracy = correct / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f} - Acc: {accuracy:.4f}")
        
        scheduler.step(epoch_val_loss)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == "__main__":
    train_model()