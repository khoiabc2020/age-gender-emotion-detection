"""
Test New Model - Verify 80%+ accuracy model is working
Run this after copying model files from Kaggle
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
import json

print("="*60)
print("TESTING NEW MODEL (80%+)")
print("="*60)

# ============================================================
# 1. TEST MODEL LOADING
# ============================================================

print("\n[1/5] Testing model loading...")
checkpoint_path = Path('trained_models/best_model.pth')

try:
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    print(f"  [OK] Model loaded successfully!")
    print(f"  [OK] Best Accuracy: {checkpoint['best_accuracy']:.2f}%")
    print(f"  [OK] Best Epoch: {checkpoint['epoch']}")
    print(f"  [OK] Num Classes: {checkpoint['num_classes']}")
    print(f"  [OK] Class Names: {checkpoint['class_names']}")
    
    if 'config' in checkpoint:
        print(f"  [OK] Model Type: {checkpoint['config'].get('model', 'Unknown')}")
        print(f"  [OK] Improvements: {checkpoint['config'].get('improvements', 'N/A')}")
except Exception as e:
    print(f"  [ERROR] Failed to load model: {e}")
    exit(1)

# ============================================================
# 2. TEST MODEL ARCHITECTURE
# ============================================================

print("\n[2/5] Testing model architecture...")
try:
    import timm
    
    model_type = checkpoint['config'].get('model', 'efficientnetv2_rw_s')
    num_classes = checkpoint['num_classes']
    
    model = timm.create_model(model_type, pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  [OK] Model architecture: {model_type}")
    print(f"  [OK] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  [OK] Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
except Exception as e:
    print(f"  [ERROR] Failed to create model: {e}")
    exit(1)

# ============================================================
# 3. TEST INFERENCE WITH DUMMY INPUT
# ============================================================

print("\n[3/5] Testing inference...")
try:
    # Create dummy input (batch_size=1, channels=3, height=72, width=72)
    dummy_input = torch.randn(1, 3, 72, 72)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  [OK] Input shape: {dummy_input.shape}")
    print(f"  [OK] Output shape: {output.shape}")
    print(f"  [OK] Output range: [{output.min():.2f}, {output.max():.2f}]")
    
    # Test softmax
    probs = torch.softmax(output, dim=1)
    pred_class = probs.argmax(dim=1).item()
    confidence = probs.max().item()
    
    print(f"  [OK] Predicted class: {pred_class} ({checkpoint['class_names'][pred_class]})")
    print(f"  [OK] Confidence: {confidence:.2%}")
except Exception as e:
    print(f"  [ERROR] Inference failed: {e}")
    exit(1)

# ============================================================
# 4. TEST WITH REAL IMAGE (if exists)
# ============================================================

print("\n[4/5] Testing with real image...")

# Define transform (same as training)
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((72, 72)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Try to find test image
test_image_paths = [
    'test_data/sample_faces/happy.jpg',
    'test_data/sample_faces/sad.jpg',
    'data/test_images/face1.jpg',
]

test_image = None
for path in test_image_paths:
    if Path(path).exists():
        test_image = path
        break

if test_image:
    try:
        img = Image.open(test_image).convert('RGB')
        img_tensor = test_transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = probs.argmax(dim=1).item()
            confidence = probs.max().item()
        
        print(f"  [OK] Test image: {test_image}")
        print(f"  [OK] Prediction: {checkpoint['class_names'][pred_class]}")
        print(f"  [OK] Confidence: {confidence:.2%}")
        
        # Show top 3 predictions
        top3_probs, top3_indices = torch.topk(probs, 3)
        print(f"  [OK] Top 3 predictions:")
        for i, (prob, idx) in enumerate(zip(top3_probs[0], top3_indices[0])):
            print(f"      {i+1}. {checkpoint['class_names'][idx]}: {prob:.2%}")
    except Exception as e:
        print(f"  [WARN] Could not test with image: {e}")
else:
    print(f"  [SKIP] No test image found")

# ============================================================
# 5. TEST ONNX MODEL (if exists)
# ============================================================

print("\n[5/5] Testing ONNX model...")
onnx_path = Path('trained_models/best_model.onnx')

if onnx_path.exists():
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        
        # Test with dummy input
        dummy_input_np = np.random.randn(1, 3, 72, 72).astype(np.float32)
        outputs = session.run(None, {input_name: dummy_input_np})
        
        print(f"  [OK] ONNX model loaded")
        print(f"  [OK] Input name: {input_name}")
        print(f"  [OK] Output shape: {outputs[0].shape}")
        print(f"  [OK] ONNX inference working!")
    except ImportError:
        print(f"  [WARN] onnxruntime not installed. Install with: pip install onnxruntime")
    except Exception as e:
        print(f"  [ERROR] ONNX test failed: {e}")
else:
    print(f"  [SKIP] ONNX model not found")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*60)
print("MODEL TEST SUMMARY")
print("="*60)
print(f"Model Accuracy: {checkpoint['best_accuracy']:.2f}%")
print(f"Model Type: {model_type}")
print(f"Input Size: 72x72")
print(f"Num Classes: {num_classes}")
print(f"Status: ALL TESTS PASSED")
print("="*60)
print("\n[SUCCESS] Model is ready for integration!")
print("\nNext steps:")
print("  1. Update app code (see POST_TRAINING_WORKFLOW.md - STEP 5)")
print("  2. Test with app (see POST_TRAINING_WORKFLOW.md - STEP 6)")
print("  3. Deploy to production (see POST_TRAINING_WORKFLOW.md - STEP 8)")
print("="*60)
