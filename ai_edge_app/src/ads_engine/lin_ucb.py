import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import os

class LinUCB:
    """
    LinUCB (Linear Upper Confidence Bound) algorithm for Contextual Bandits.
    Used for selecting the best ad based on user context (Age, Gender, Emotion).
    """
    
    def __init__(self, alpha: float = 0.1, n_features: int = 20, model_path: str = "models/linucb_weights.pkl"):
        """
        Initialize LinUCB
        
        Args:
            alpha: Exploration parameter (higher = more exploration)
            n_features: Number of context features
            model_path: Path to save/load model weights
        """
        self.alpha = alpha
        self.n_features = n_features
        self.model_path = model_path
        
        # Initialize dictionary to store A and b for each arm (ad_id)
        # A: Design matrix covariance (d x d)
        # b: Reward vector (d x 1)
        self.arms = {}
        
        # Load existing weights if available
        self.load_weights()
        
    def _init_arm(self, arm_id: str):
        """Initialize matrices for a new arm"""
        if arm_id not in self.arms:
            self.arms[arm_id] = {
                'A': np.identity(self.n_features),
                'b': np.zeros((self.n_features, 1))
            }
            
    def get_prediction(self, context: np.ndarray, arm_id: str) -> float:
        """
        Calculate UCB score for a specific arm
        
        Args:
            context: Feature vector of current user
            arm_id: Ad ID
            
        Returns:
            UCB score (exploitation + exploration)
        """
        self._init_arm(arm_id)
        
        arm = self.arms[arm_id]
        A = arm['A']
        b = arm['b']
        
        # Calculate inverse of A
        # Optimization: storing A_inv and updating it via Sherman-Morrison is faster, 
        # but for small dimensions np.linalg.inv is fine
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A)
            
        theta = A_inv @ b
        
        # Prediction (Mean)
        p = theta.T @ context
        
        # Confidence Interval (Variance)
        c = self.alpha * np.sqrt(context.T @ A_inv @ context)
        
        return float(p + c)
        
    def select_arm(self, context: np.ndarray, available_arms: List[str]) -> str:
        """
        Select the best arm from available options
        
        Args:
            context: Feature vector
            available_arms: List of valid Ad IDs
            
        Returns:
            Selected Ad ID
        """
        best_arm = None
        max_ucb = -float('inf')
        
        # Reshape context to column vector (d, 1) if needed
        if context.ndim == 1:
            context = context.reshape(-1, 1)
            
        for arm_id in available_arms:
            ucb = self.get_prediction(context, arm_id)
            if ucb > max_ucb:
                max_ucb = ucb
                best_arm = arm_id
                
        # Fallback if list is empty (should not happen in logic)
        if best_arm is None and available_arms:
            return available_arms[0]
            
        return best_arm
        
    def update(self, context: np.ndarray, arm_id: str, reward: float):
        """
        Update the model with feedback
        
        Args:
            context: Feature vector used at selection time
            arm_id: Selected Ad ID
            reward: Observed reward (e.g., viewing time)
        """
        self._init_arm(arm_id)
        
        # Reshape context
        if context.ndim == 1:
            context = context.reshape(-1, 1)
            
        # Update A and b
        self.arms[arm_id]['A'] += context @ context.T
        self.arms[arm_id]['b'] += reward * context
        
        # Auto-save periodically could be added here, but explicit save is better
        self.save_weights()
        
    def save_weights(self):
        """Save model weights"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.arms, f)
        except Exception as e:
            print(f"Failed to save LinUCB weights: {e}")
            
    def load_weights(self):
        """Load model weights"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.arms = pickle.load(f)
            except Exception as e:
                print(f"Failed to load LinUCB weights: {e}")
