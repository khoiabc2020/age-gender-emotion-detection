"""
Knowledge Distillation for Model Compression
Teacher: ResNet50 (large model)
Student: MobileOne-S2 (lightweight model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss
    Combines hard target loss (ground truth) and soft target loss (teacher predictions)
    """
    def __init__(self, temperature=4.0, alpha=0.7):
        """
        Args:
            temperature: Temperature for softmax (higher = softer distribution)
            alpha: Weight for soft target loss (1-alpha for hard target)
        """
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, hard_targets):
        """
        Args:
            student_logits: Student model predictions (B, C)
            teacher_logits: Teacher model predictions (B, C)
            hard_targets: Ground truth labels (B,)
        """
        # Soft target loss (KL divergence between teacher and student)
        soft_loss = self.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        
        # Hard target loss (cross entropy with ground truth)
        hard_loss = self.ce_loss(student_logits, hard_targets)
        
        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return {
            'total': total_loss,
            'soft': soft_loss,
            'hard': hard_loss
        }


class TeacherModel(nn.Module):
    """
    Teacher Model: ResNet50
    Large model used to teach the student
    """
    def __init__(self, num_emotions=6):
        super(TeacherModel, self).__init__()
        
        # ResNet50 backbone
        self.backbone = models.resnet50(weights='IMAGENET1K_V2')
        feature_dim = self.backbone.fc.in_features
        
        # Remove original classifier
        self.backbone.fc = nn.Identity()
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Multi-task heads
        self.gender_head = nn.Linear(512, 2)
        self.age_head = nn.Linear(512, 1)
        self.emotion_head = nn.Linear(512, num_emotions)
    
    def forward(self, x):
        # Extract features
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Project features
        features = self.feature_projection(x)
        
        # Multi-task heads
        gender_logits = self.gender_head(features)
        age_pred = self.age_head(features)
        emotion_logits = self.emotion_head(features)
        
        age_pred = torch.clamp(age_pred, min=0.0, max=100.0)
        
        return gender_logits, age_pred, emotion_logits


class MultiTaskDistillationLoss(nn.Module):
    """
    Multi-task Knowledge Distillation Loss
    Applies distillation to all three tasks: Gender, Age, Emotion
    """
    def __init__(self, temperature=4.0, alpha=0.7, age_weight=0.5, emotion_weight=1.0):
        super(MultiTaskDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.age_weight = age_weight
        self.emotion_weight = emotion_weight
        
        # Distillation losses for each task
        self.gender_distill = DistillationLoss(temperature, alpha)
        self.emotion_distill = DistillationLoss(temperature, alpha)
        
        # Age uses MSE for regression
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_preds, teacher_preds, targets):
        """
        Args:
            student_preds: (gender_logits, age_pred, emotion_logits)
            teacher_preds: (gender_logits, age_pred, emotion_logits)
            targets: (gender_target, age_target, emotion_target)
        """
        student_gender, student_age, student_emotion = student_preds
        teacher_gender, teacher_age, teacher_emotion = teacher_preds
        gender_target, age_target, emotion_target = targets
        
        # Gender distillation
        gender_loss_dict = self.gender_distill(student_gender, teacher_gender, gender_target)
        
        # Age loss (MSE for regression)
        age_soft_loss = F.mse_loss(
            student_age / self.temperature,
            teacher_age / self.temperature
        ) * (self.temperature ** 2)
        age_hard_loss = self.mse_loss(student_age.squeeze(), age_target.float())
        age_loss = self.alpha * age_soft_loss + (1 - self.alpha) * age_hard_loss
        
        # Emotion distillation
        emotion_loss_dict = self.emotion_distill(student_emotion, teacher_emotion, emotion_target)
        
        # Total loss
        total_loss = (
            gender_loss_dict['total'] +
            self.age_weight * age_loss +
            self.emotion_weight * emotion_loss_dict['total']
        )
        
        return {
            'total': total_loss,
            'gender': gender_loss_dict['total'],
            'age': age_loss,
            'emotion': emotion_loss_dict['total'],
            'gender_soft': gender_loss_dict['soft'],
            'gender_hard': gender_loss_dict['hard'],
            'emotion_soft': emotion_loss_dict['soft'],
            'emotion_hard': emotion_loss_dict['hard']
        }






