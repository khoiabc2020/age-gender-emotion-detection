"""
MobileOne-S2 Architecture for Edge AI
Lightweight SOTA model for mobile/edge devices
Based on: https://github.com/apple/ml-mobileone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MobileOneBlock(nn.Module):
    """MobileOne building block"""
    def __init__(self, in_channels, out_channels, stride=1, num_conv_branches=1):
        super(MobileOneBlock, self).__init__()
        self.stride = stride
        self.num_conv_branches = num_conv_branches
        
        # Depthwise convolution
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3,
            stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)
        
        # SE block
        self.se = SEBlock(out_channels)
        
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.dw_conv(x)
        x = self.dw_bn(x)
        x = self.activation(x)
        
        x = self.pw_conv(x)
        x = self.pw_bn(x)
        x = self.se(x)
        x = self.activation(x)
        
        return x


class MobileOneS2(nn.Module):
    """
    MobileOne-S2 Backbone
    Lightweight architecture optimized for edge devices
    """
    def __init__(self, num_classes=1000, width_multiplier=1.0):
        super(MobileOneS2, self).__init__()
        
        # First layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, int(64 * width_multiplier), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(64 * width_multiplier)),
            nn.ReLU(inplace=True)
        )
        
        # MobileOne blocks
        self.stage1 = self._make_stage(int(64 * width_multiplier), int(128 * width_multiplier), 2, stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier), int(256 * width_multiplier), 4, stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier), int(512 * width_multiplier), 6, stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier), int(1024 * width_multiplier), 2, stride=2)
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Linear(int(1024 * width_multiplier), num_classes)
    
    def _make_stage(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(MobileOneBlock(in_channels, out_channels, stride=stride))
        for _ in range(num_blocks - 1):
            layers.append(MobileOneBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.first_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def extract_features(self, x):
        """Extract features before classifier"""
        x = self.first_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return x


class MobileOneMultiTaskModel(nn.Module):
    """
    MobileOne-S2 Multi-task Model
    Lightweight model for Age, Gender, and Emotion
    """
    def __init__(
        self,
        num_emotions=6,
        dropout_rate=0.3,
        width_multiplier=1.0
    ):
        super(MobileOneMultiTaskModel, self).__init__()
        
        # MobileOne-S2 backbone
        self.backbone = MobileOneS2(num_classes=1000, width_multiplier=width_multiplier)
        feature_dim = int(1024 * width_multiplier)
        
        # Remove classifier
        self.backbone.classifier = nn.Identity()
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Gender head
        self.gender_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(256, 2)
        )
        
        # Age head
        self.age_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(256, 1)
        )
        
        # Emotion head
        self.emotion_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(256, num_emotions)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone.extract_features(x)
        
        # Project features
        projected = self.feature_projection(features)
        
        # Multi-task heads
        gender_logits = self.gender_head(projected)
        age_pred = self.age_head(projected)
        emotion_logits = self.emotion_head(projected)
        
        # Clamp age
        age_pred = torch.clamp(age_pred, min=0.0, max=100.0)
        
        return gender_logits, age_pred, emotion_logits






