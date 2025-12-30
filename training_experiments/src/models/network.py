"""
Multi-task Network Architecture
Optimized với better regularization và anti-overfitting techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MultiTaskModel(nn.Module):
    """
    Multi-task Learning Model với improved regularization
    - Backbone: EfficientNet-B0 (Pre-trained on ImageNet)
    - Head 1: Gender Classification (Binary)
    - Head 2: Age Regression
    - Head 3: Emotion Classification (7 classes)
    """
    
    def __init__(
        self, 
        num_emotions=7, 
        dropout_rate=0.5,
        use_batch_norm=True,
        use_layer_norm=False
    ):
        super(MultiTaskModel, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        
        # Get feature dimension
        # EfficientNet-B0 last layer: 1280 features
        feature_dim = self.backbone.classifier[1].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Freeze early layers for better transfer learning
        # Only fine-tune last few layers
        for param in list(self.backbone.features[:-3].parameters()):
            param.requires_grad = False
        
        # Shared feature extractor với better regularization
        self.shared_features = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate)
        )
        
        # Feature projection layer để giảm overfitting
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512) if use_batch_norm else nn.LayerNorm(512) if use_layer_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8)
        )
        
        # Head 1: Gender Classification (Binary) với better architecture
        self.gender_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256) if use_batch_norm else nn.LayerNorm(256) if use_layer_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128) if use_batch_norm else nn.LayerNorm(128) if use_layer_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.4),
            nn.Linear(128, 2)  # Male, Female
        )
        
        # Head 2: Age Regression với better architecture
        self.age_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256) if use_batch_norm else nn.LayerNorm(256) if use_layer_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128) if use_batch_norm else nn.LayerNorm(128) if use_layer_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.4),
            nn.Linear(128, 1)  # Age (continuous)
        )
        
        # Head 3: Emotion Classification (7 classes) với better architecture
        self.emotion_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256) if use_batch_norm else nn.LayerNorm(256) if use_layer_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128) if use_batch_norm else nn.LayerNorm(128) if use_layer_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.4),
            nn.Linear(128, num_emotions)  # 7 emotions
        )
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor (B, 3, 224, 224)
        Returns:
            gender_logits: (B, 2)
            age_pred: (B, 1)
            emotion_logits: (B, 7)
        """
        # Extract features
        features = self.backbone.features(x)  # (B, 1280, H, W)
        features = self.shared_features(features)  # (B, 1280)
        
        # Feature projection
        projected_features = self.feature_projection(features)  # (B, 512)
        
        # Multi-task heads (parallel execution)
        gender_logits = self.gender_head(projected_features)
        age_pred = self.age_head(projected_features)
        emotion_logits = self.emotion_head(projected_features)
        
        # Ensure age prediction is non-negative and reasonable
        age_pred = torch.clamp(age_pred, min=0.0, max=100.0)
        
        return gender_logits, age_pred, emotion_logits


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    Giúp chống overfitting và cải thiện generalization
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_prob)
            true_dist.fill_(self.smoothing / (pred.size(1) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_prob, dim=1))


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
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


class MultiTaskLoss(nn.Module):
    """
    Combined Loss Function với advanced techniques
    - Label Smoothing cho classification tasks
    - Focal Loss option cho imbalanced data
    - Weighted loss với trọng số có thể điều chỉnh
    """
    
    def __init__(
        self, 
        age_weight=0.5, 
        gender_weight=1.0, 
        emotion_weight=1.0,
        label_smoothing=0.1,
        use_focal_loss=False,
        focal_alpha=1.0,
        focal_gamma=2.0
    ):
        super(MultiTaskLoss, self).__init__()
        self.age_weight = age_weight
        self.gender_weight = gender_weight
        self.emotion_weight = emotion_weight
        self.label_smoothing = label_smoothing
        self.use_focal_loss = use_focal_loss
        
        # Loss functions với label smoothing
        if label_smoothing > 0:
            self.gender_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
            self.emotion_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        elif use_focal_loss:
            self.gender_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            self.emotion_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.gender_loss = nn.CrossEntropyLoss()
            self.emotion_loss = nn.CrossEntropyLoss()
        
        # Age loss: L1 (MAE) hoặc Huber Loss (smooth L1)
        # Huber Loss ít nhạy với outliers hơn
        self.age_loss = nn.SmoothL1Loss()  # Huber Loss (beta=1.0)
        # Alternative: nn.L1Loss() for pure MAE
    
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
        loss_age = self.age_loss(age_pred.squeeze(), age_target)
        loss_emotion = self.emotion_loss(emotion_logits, emotion_target)
        
        # Combined loss với trọng số
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
    print("Testing Optimized MultiTaskModel...")
    
    model = MultiTaskModel()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    gender_logits, age_pred, emotion_logits = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Gender logits shape: {gender_logits.shape}")
    print(f"Age prediction shape: {age_pred.shape}")
    print(f"Emotion logits shape: {emotion_logits.shape}")
    
    # Test loss với label smoothing
    criterion = MultiTaskLoss(label_smoothing=0.1)
    targets = (
        torch.randint(0, 2, (2,)),  # gender
        torch.randn(2) * 20 + 30,   # age
        torch.randint(0, 7, (2,))   # emotion
    )
    
    predictions = (gender_logits, age_pred, emotion_logits)
    losses = criterion(predictions, targets)
    
    print(f"\nLosses:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
