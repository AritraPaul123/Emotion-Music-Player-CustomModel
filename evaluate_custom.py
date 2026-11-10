import os
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from custom_model import EmotionCNN

EMOTION_MAP = {
    'neutral': 0,
    'happiness': 1,
    'surprise': 2,
    'sadness': 3,
    'anger': 4
}
CLASS_NAMES = ['neutral', 'happy', 'surprise', 'sad', 'angry']

def evaluate():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'dataset')
    manifest_path = os.path.join(data_dir, 'test_manifest.csv')
    img_dir = os.path.join(data_dir, 'images')
    weights_path = os.path.join(base_dir, 'custom_model_weights.pth')
    
    df = pd.read_csv(manifest_path)
    print(f"Loaded test set with {len(df)} samples.")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = EmotionCNN(num_classes=5).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    val_transforms = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    y_true = []
    y_pred = []
    latencies = []
    
    print("Running Custom Model evaluation on test set...")
    with torch.no_grad():
        for idx, row in df.iterrows():
            img_path = os.path.join(img_dir, row['image'])
            true_label = row['label']
            
            try:
                img = Image.open(img_path).convert('L')
                t_img = val_transforms(img).unsqueeze(0).to(device)
                
                start_t = time.perf_counter()
                output = model(t_img)
                pred_label = torch.argmax(output, dim=1).item()
                latencies.append((time.perf_counter() - start_t) * 1000) # ms
                
                y_true.append(true_label)
                y_pred.append(pred_label)
            except Exception as e:
                continue
                
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    avg_latency = float(np.mean(latencies))
    
    results = {
        'model': 'Custom Emotion CNN (PyTorch)',
        'samples': len(y_true),
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'avg_latency_ms': avg_latency,
        'confusion_matrix': cm,
        'classes': CLASS_NAMES,
        'classification_report': classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    }
    
    out_path = os.path.join(base_dir, 'custom_model_eval_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Custom Model Accuracy: {acc*100:.2f}%")
    print(f"Weighted F1: {f1:.4f}")
    print(f"Average Inference Latency: {avg_latency:.2f} ms")
    print(f"Results saved to {out_path}")

if __name__ == '__main__':
    evaluate()

