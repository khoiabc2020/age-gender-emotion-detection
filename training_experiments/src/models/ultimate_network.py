"""
Ultimate Multi-task Network Architecture - SOTA Edition
SOTA Model Architecture
- CBAM Attention Mechanism
- CORAL for Age Estimation
- Focal Loss & Wing Loss
- 6 Emotion Classes (Disgust merged into Angry)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out

class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return x * out

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    Giúp model "tập trung" vào mắt/miệng thay vì nền ảnh
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class CORALAgeHead(nn.Module):
    """
    CORAL (Consistent Rank Logits) for Age Estimation
    Thay vì regression thường, dùng ordinal regression
    Giảm sai số MAE đáng kể
    """
    def __init__(self, feature_dim=512, num_classes=101):
        """
        Args:
            feature_dim: Input feature dimension
            num_classes: Number of age classes (0-100)
        """
        super(CORALAgeHead, self).__init__()
        self.num_classes = num_classes
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # CORAL: num_classes - 1 binary classifiers
        self.classifiers = nn.ModuleList([
            nn.Linear(128, 1) for _ in range(num_classes - 1)
        ])
    
    def forward(self, x):
        """
        Args:
            x: (B, feature_dim)
        Returns:
            logits: (B, num_classes - 1) - Binary logits for each threshold
        """
        features = self.shared(x)
        logits = torch.stack([classifier(features) for classifier in self.classifiers], dim=1)
        return logits.squeeze(-1)  # (B, num_classes - 1)
    
    def predict_age(self, logits):
        """
        Convert CORAL logits to age prediction
        Args:
            logits: (B, num_classes - 1)
        Returns:
            age: (B,) - Predicted age
        """
        # Convert binary logits to probabilities
        probs = torch.sigmoid(logits)  # (B, num_classes - 1)
        
        # Sum probabilities to get predicted age
        # Each threshold represents "age > threshold"
        age_pred = torch.sum(probs, dim=1)  # (B,)
        
        return age_pred

class UltimateMultiTaskModel(nn.Module):
    """
    Ultimate Multi-task Learning Model với CBAM Attention
    - Backbone: EfficientNet-B0 (Pre-trained)
    - CBAM Attention: Tập trung vào features quan trọng
    - CORAL Age Head: Ordinal regression cho age
    - 6 Emotion Classes (Disgust merged into Angry)
    """
    
    def __init__(
        self,
        num_emotions=6,  # 6 classes sau khi gộp Disgust vào Angry
        dropout_rate=0.5,
        use_cbam=True,
        use_coral=True,
        num_age_classes=101  # 0-100 years
    ):
        super(UltimateMultiTaskModel, self).__init__()
        
        self.use_cbam = use_cbam
        self.use_coral = use_coral
        self._age_logits = None  # Store CORAL logits
        
        # Load pre-trained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        
        # Get feature dimension
        feature_dim = self.backbone.classifier[1].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Freeze early layers
        for param in list(self.backbone.features[:-3].parameters()):
            param.requires_grad = False
        
        # Add CBAM attention after backbone features
        if use_cbam:
            # Get channel dimension from last EfficientNet block
            # EfficientNet-B0 last block output: 1280 channels
            self.cbam = CBAM(in_channels=1280, reduction=16)
        
        # Shared feature extractor
        self.shared_features = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate)
        )
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),  # Swish activation
            nn.Dropout(dropout_rate * 0.8)
        )
        
        # Head 1: Gender Classification (Binary)
        self.gender_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(256, 2)  # Male, Female
        )
        
        # Head 2: Age Estimation
        if use_coral:
            # CORAL: Ordinal regression
            self.age_head = CORALAgeHead(feature_dim=512, num_classes=num_age_classes)
        else:
            # Standard regression
            self.age_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout_rate * 0.6),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout_rate * 0.4),
                nn.Linear(128, 1)
            )
        
        # Head 3: Emotion Classification (6 classes)
        self.emotion_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(256, num_emotions)  # 6 emotions
        )
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor (B, 3, 224, 224)
        Returns:
            gender_logits: (B, 2)
            age_pred: (B, 1) or (B, num_classes-1) for CORAL
            emotion_logits: (B, 6)
        """
        # Extract features
        features = self.backbone.features(x)  # (B, 1280, H, W)
        
        # Apply CBAM attention
        if self.use_cbam:
            features = self.cbam(features)
        
        # Global pooling
        features = self.shared_features(features)  # (B, 1280)
        
        # Feature projection
        projected_features = self.feature_projection(features)  # (B, 512)
        
        # Multi-task heads
        gender_logits = self.gender_head(projected_features)
        
        if self.use_coral:
            age_logits = self.age_head(projected_features)  # (B, num_classes-1)
            # Store logits for CORAL loss
            self._age_logits = age_logits
            # Convert to age prediction
            age_pred = self.age_head.predict_age(age_logits).unsqueeze(1)  # (B, 1)
        else:
            age_pred = self.age_head(projected_features)  # (B, 1)
        
        emotion_logits = self.emotion_head(projected_features)
        
        # Clamp age to valid range
        age_pred = torch.clamp(age_pred, min=0.0, max=100.0)
        
        return gender_logits, age_pred, emotion_logits

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focus on hard examples
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WingLoss(nn.Module):
    """
    Wing Loss for Age Regression
    Better than MSE for age estimation
    Less sensitive to outliers
    """
    def __init__(self, w=10.0, eps=2.0):
        super(WingLoss, self).__init__()
        self.w = w
        self.eps = eps
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1) or (B,)
            target: (B, 1) or (B,)
        """
        diff = torch.abs(pred.squeeze() - target.squeeze())
        
        # Wing loss formula
        loss = torch.where(
            diff < self.w,
            self.w * torch.log(1 + diff / self.eps),
            diff - self.w + self.w * torch.log(1 + self.w / self.eps)
        )
        
        return loss.mean()

class CORALLoss(nn.Module):
    """
    CORAL Loss for Ordinal Regression
    Used with CORAL Age Head
    """
    def __init__(self, num_classes=101):
        super(CORALLoss, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, num_classes - 1) - Binary logits
            targets: (B,) - Age labels (0-100)
        """
        # Convert age to binary labels
        # For age y, labels should be [1, 1, ..., 1, 0, 0, ..., 0]
        # where first y labels are 1
        targets = targets.long()
        
        # Create binary targets
        binary_targets = torch.zeros_like(logits)
        for i in range(self.num_classes - 1):
            binary_targets[:, i] = (targets > i).float()
        
        # Binary cross entropy loss
        loss = F.binary_cross_entropy_with_logits(logits, binary_targets, reduction='mean')
        
        return loss

class UltimateMultiTaskLoss(nn.Module):
    """
    Combined Loss Function với Focal Loss và Wing Loss
    """
    
    def __init__(
        self,
        age_weight=0.5,
        gender_weight=1.0,
        emotion_weight=1.0,
        use_focal_loss=True,
        use_wing_loss=True,
        use_coral=False,
        focal_alpha=1.0,
        focal_gamma=2.0,
        num_age_classes=101
    ):
        super(UltimateMultiTaskLoss, self).__init__()
        self.age_weight = age_weight
        self.gender_weight = gender_weight
        self.emotion_weight = emotion_weight
        self.use_focal_loss = use_focal_loss
        self.use_wing_loss = use_wing_loss
        self.use_coral = use_coral
        
        # Loss functions
        if use_focal_loss:
            self.gender_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            self.emotion_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.gender_loss = nn.CrossEntropyLoss()
            self.emotion_loss = nn.CrossEntropyLoss()
        
        if use_coral:
            self.age_loss = CORALLoss(num_classes=num_age_classes)
        elif use_wing_loss:
            self.age_loss = WingLoss(w=10.0, eps=2.0)
        else:
            self.age_loss = nn.SmoothL1Loss()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (gender_logits, age_pred, emotion_logits)
            targets: (gender, age, emotion)
        """
        gender_logits, age_pred, emotion_logits = predictions
        gender_target, age_target, emotion_target = targets
        
        # Calculate losses
        loss_gender = self.gender_loss(gender_logits, gender_target)
        
        if self.use_coral:
            # For CORAL, age_pred should be logits (B, num_classes-1) from training loop
            # Check if it's logits (shape > 1) or predicted age (shape = 1)
            if age_pred.dim() > 1 and age_pred.shape[1] > 1:
                # This is logits (CORAL format)
                loss_age = self.age_loss(age_pred, age_target)
            else:
                # Fallback: if somehow we got predicted age instead of logits
                import warnings
                warnings.warn("CORAL: Expected logits but got predicted age, using fallback")
                loss_age = nn.SmoothL1Loss()(age_pred.squeeze(), age_target)
        else:
            # Standard regression: age_pred is (B, 1) or (B,)
            loss_age = self.age_loss(age_pred.squeeze(), age_target)
        
        loss_emotion = self.emotion_loss(emotion_logits, emotion_target)
        
        # Combined loss
        total_loss = (
            self.gender_weight * loss_gender +
            self.age_weight * loss_age +
            self.emotion_weight * loss_emotion
        )
        
        return {
            'total': total_loss,
            'gender': loss_gender,
            'age': loss_age,
            'emotion': loss_emotion
        }

if __name__ == "__main__":
    # Test model
    print("Testing Ultimate MultiTaskModel...")
    
    model = UltimateMultiTaskModel(num_emotions=6, use_cbam=True, use_coral=True)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    gender_logits, age_pred, emotion_logits = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Gender logits shape: {gender_logits.shape}")
    print(f"Age prediction shape: {age_pred.shape}")
    print(f"Emotion logits shape: {emotion_logits.shape}")
    
    # Test loss
    criterion = UltimateMultiTaskLoss(use_focal_loss=True, use_wing_loss=True, use_coral=True)
    targets = (
        torch.randint(0, 2, (2,)),  # gender
        torch.randint(0, 101, (2,)).float(),  # age
        torch.randint(0, 6, (2,))  # emotion (6 classes)
    )
    
    predictions = (gender_logits, age_pred, emotion_logits)
    losses = criterion(predictions, targets)
    
    print(f"\nLosses:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")

