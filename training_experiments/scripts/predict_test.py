"""
Test Inference Script
Test model với ảnh bất kỳ
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import onnxruntime as ort
import numpy as np
from pathlib import Path

def predict_pytorch(model_path, image_path, device='cpu'):
    """Predict using PyTorch model"""
    # Load model
    from src.models.network import MultiTaskModel
    
    model = MultiTaskModel()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        gender_logits, age_pred, emotion_logits = model(input_tensor)
    
    # Process results
    gender = gender_logits.argmax(dim=1).item()
    age = age_pred.item()
    emotion = emotion_logits.argmax(dim=1).item()
    
    gender_names = ['Male', 'Female']
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    return {
        'gender': gender_names[gender],
        'age': int(round(age)),
        'emotion': emotion_names[emotion]
    }

def predict_onnx(model_path, image_path):
    """Predict using ONNX model"""
    # Load ONNX model
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    
    # Convert to numpy array and normalize
    image_array = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array - mean) / std
    
    # Convert to CHW format
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)
    
    # Predict
    outputs = session.run(None, {input_name: image_array})
    
    gender_logits, age_pred, emotion_logits = outputs
    
    # Process results
    gender = np.argmax(gender_logits, axis=1)[0]
    age = age_pred[0][0]
    emotion = np.argmax(emotion_logits, axis=1)[0]
    
    gender_names = ['Male', 'Female']
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    return {
        'gender': gender_names[gender],
        'age': int(round(age)),
        'emotion': emotion_names[emotion]
    }

def main():
    parser = argparse.ArgumentParser(description='Test model inference')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model (.pth or .onnx)')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to test image')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return
    
    # Check if image exists
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        return
    
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print("=" * 60)
    
    # Predict
    if model_path.suffix == '.onnx':
        print("Using ONNX model...")
        results = predict_onnx(str(model_path), str(image_path))
    else:
        print("Using PyTorch model...")
        results = predict_pytorch(str(model_path), str(image_path), args.device)
    
    # Print results
    print("\n[INFO] Prediction Results:")
    print(f"  Gender: {results['gender']}")
    print(f"  Age: {results['age']} years")
    print(f"  Emotion: {results['emotion']}")
    print("=" * 60)

if __name__ == "__main__":
    main()

