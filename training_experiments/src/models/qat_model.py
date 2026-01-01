"""
Quantization-Aware Training (QAT) Model Wrapper
Enables quantization during training for better edge deployment
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.quantization import QuantStub, DeQuantStub

class QATMultiTaskModel(nn.Module):
    """
    Quantization-Aware Training Wrapper
    Wraps any multi-task model for QAT
    """
    def __init__(self, base_model):
        """
        Args:
            base_model: Base multi-task model (MobileOne, EfficientNet, etc.)
        """
        super(QATMultiTaskModel, self).__init__()
        self.base_model = base_model
        
        # Quantization stubs
        self.quant = QuantStub()
        self.dequant_gender = DeQuantStub()
        self.dequant_age = DeQuantStub()
        self.dequant_emotion = DeQuantStub()
    
    def forward(self, x):
        # Quantize input
        x = self.quant(x)
        
        # Forward through base model
        gender_logits, age_pred, emotion_logits = self.base_model(x)
        
        # Dequantize outputs
        gender_logits = self.dequant_gender(gender_logits)
        age_pred = self.dequant_age(age_pred)
        emotion_logits = self.dequant_emotion(emotion_logits)
        
        return gender_logits, age_pred, emotion_logits
    
    def prepare_qat(self):
        """Prepare model for QAT"""
        quantization.prepare_qat(self, inplace=True)
    
    def convert_to_quantized(self):
        """Convert trained QAT model to quantized model"""
        self.eval()
        quantization.convert(self, inplace=True)
        return self

