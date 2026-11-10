import os
import json
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from custom_model import EmotionCNN

# Standardize emotions to align with facial expressions
EMOTION_MAP = {
    'neutral': 0,
    'happiness': 1,
    'surprise': 2,
    'sadness': 3,
    'anger': 4
}
CLASS_NAMES = ['neutral', 'happy', 'surprise', 'sad', 'angry']

class FacialDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image']
        label = row['label']
        
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('L') # Convert to grayscale
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def prepare_data(data_dir):
    legend_path = os.path.join(data_dir, 'legend.csv')
    img_dir = os.path.join(data_dir, 'images')
    
    df = pd.read_csv(legend_path)
    df['emotion_clean'] = df['emotion'].str.lower().str.strip()
    
    # Filter only supported classes
    df = df[df['emotion_clean'].isin(EMOTION_MAP.keys())].copy()
    df['label'] = df['emotion_clean'].map(EMOTION_MAP)
    
    # Verify file existence
    valid_mask = df['image'].apply(lambda x: os.path.exists(os.path.join(img_dir, x)))
    df = df[valid_mask]
    
    # Stratified train/val/test split (70% train, 15% val, 15% test)
    train_df, test_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(test_df, test_size=0.50, random_state=42, stratify=test_df['label'])
    
    print(f"Dataset split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Save test set manifest so both Custom and DeepFace evaluate on identical samples
    test_manifest_path = os.path.join(data_dir, 'test_manifest.csv')
    test_df.to_csv(test_manifest_path, index=False)
    print(f"Saved test manifest to {test_manifest_path}")
    
    return train_df, val_df, test_df, img_dir

def train():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    data_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    train_df, val_df, test_df, img_dir = prepare_data(data_dir)
    
    # Image augmentations & normalization
    train_transforms = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_dataset = FacialDataset(train_df, img_dir, transform=train_transforms)
    val_dataset = FacialDataset(val_df, img_dir, transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    # Compute class weights for handling class imbalance
    class_counts = train_df['label'].value_counts().sort_index().values
    total_samples = len(train_df)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights_t = torch.FloatTensor(class_weights).to(device)
    
    model = EmotionCNN(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    epochs = 12
    best_val_acc = 0.0
    model_save_path = os.path.join(os.path.dirname(__file__), 'custom_model_weights.pth')
    
    print("\nStarting training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Best model saved to {model_save_path} (Val Acc: {val_acc*100:.2f}%)")
            
    print("\nTraining completed successfully!")

if __name__ == '__main__':
    train()

