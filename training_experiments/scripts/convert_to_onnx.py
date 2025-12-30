"""
Script to convert PyTorch model to ONNX format
Tuần 4: Chuyển đổi model sang ONNX cho Edge Computing
"""

import torch
import onnx
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.network import MultiTaskModel


def convert_to_onnx(
    model_path: str,
    output_path: str,
    input_size: tuple = (1, 3, 224, 224),
    opset_version: int = 13  # Updated to opset 13+ for Tuần 2
):
    """
    Convert PyTorch model to ONNX format
    
    Args:
        model_path: Path to PyTorch model checkpoint (.pth)
        output_path: Output ONNX file path
        input_size: Input tensor size (batch, channels, height, width)
        opset_version: ONNX opset version
    """
    print("=" * 60)
    print("Converting PyTorch model to ONNX")
    print("=" * 60)
    
    # Load model
    device = torch.device('cpu')
    print(f"Loading model from: {model_path}")
    
    model = MultiTaskModel()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy input
    dummy_input = torch.randn(*input_size)
    print(f"Input shape: {dummy_input.shape}")
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['gender', 'age', 'emotion'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'gender': {0: 'batch_size'},
            'age': {0: 'batch_size'},
            'emotion': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"✅ Model exported to: {output_path}")
    
    # Verify ONNX model
    print("\nVerifying ONNX model...")
    try:
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model is valid!")
        
        # Print model info
        print(f"\nModel info:")
        print(f"  Input: {onnx_model.graph.input[0].name}")
        print(f"  Outputs: {[out.name for out in onnx_model.graph.output]}")
        print(f"  Opset version: {onnx_model.opset_import[0].version}")
        
    except Exception as e:
        print(f"❌ Error verifying ONNX model: {e}")
        return False
    
    # Test inference
    print("\nTesting ONNX inference...")
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(output_path))
        input_name = session.get_inputs()[0].name
        
        # Test with dummy input
        dummy_np = dummy_input.numpy()
        outputs = session.run(None, {input_name: dummy_np})
        
        print(f"✅ ONNX inference successful!")
        print(f"  Gender logits shape: {outputs[0].shape}")
        print(f"  Age prediction shape: {outputs[1].shape}")
        print(f"  Emotion logits shape: {outputs[2].shape}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not test ONNX inference: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Conversion completed successfully!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument("--model_path", type=str, required=True, help="Path to PyTorch checkpoint (.pth)")
    parser.add_argument("--output_path", type=str, required=True, help="Output ONNX file path")
    parser.add_argument("--input_size", type=int, nargs=4, default=[1, 3, 224, 224], help="Input size (batch, channels, height, width)")
    parser.add_argument("--opset_version", type=int, default=13, help="ONNX opset version (13+ for Tuần 2)")
    
    args = parser.parse_args()
    
    success = convert_to_onnx(
        model_path=args.model_path,
        output_path=args.output_path,
        input_size=tuple(args.input_size),
        opset_version=args.opset_version
    )
    
    if not success:
        sys.exit(1)

