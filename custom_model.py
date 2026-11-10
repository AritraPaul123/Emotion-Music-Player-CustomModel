import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    """
    Custom Convolutional Neural Network for Facial Emotion Recognition.
    Architecture:
    - 4 Convolutional blocks with BatchNorm, ReLU, and MaxPool
    - Dropout for regularization
    - Fully Connected classification head
    """
    def __init__(self, num_classes=5):
        super(EmotionCNN, self).__init__()
        
        # Block 1: 1 x 48 x 48 -> 32 x 24 x 24
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Block 2: 32 x 24 x 24 -> 64 x 12 x 12
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 3: 64 x 12 x 12 -> 128 x 6 x 6
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Block 4: 128 x 6 x 6 -> 256 x 3 x 3
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.25)
        self.dropout_fc = nn.Dropout(0.5)
        
        # FC Head
        self.fc1 = nn.Linear(256 * 3 * 3, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # Block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

