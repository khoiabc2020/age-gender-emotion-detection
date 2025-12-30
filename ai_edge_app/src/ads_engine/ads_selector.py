"""
Hybrid Advertisement Selection Engine
Implements filtering, scoring, and exploration strategies
"""

import json
import random
from typing import Dict, List, Optional
from datetime import datetime



from .lin_ucb import LinUCB

class AdsSelector:
    """
    Advertisement selector using hybrid approach:
    1. Filtering: Remove unsuitable ads (Safety Rules)
    2. Reinforcement Learning: LinUCB for optimal selection
    """
    
    def __init__(self, rules_path: str):
        """
        Initialize ads selector
        """
        with open(rules_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.ad_rules = config['ad_rules']
        self.fallback_ad = config.get('fallback_ad', 'default_promotion')
        self.display_history = {}  # Track ad display history
        
        # RL Model
        # Features: Age (1) + Gender (2) + Emotion (7) + Time (1) = 11 features approx
        self.linucb = LinUCB(n_features=11)
        
        # Map emotions to indices for one-hot encoding
        self.emotion_map = {
            'angry': 0, 'fear': 1, 'neutral': 2, 'happy': 3, 
            'sad': 4, 'surprise': 5, 'disgust': 6
        }
    
    def select_ad(
        self, 
        age: int, 
        gender: str, 
        emotion: str,
        track_id: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Select best advertisement based on customer attributes
        """
        # Step 1: Filtering (Safety Layer)
        filtered_ads = self._filter_ads(age, gender, emotion)
        
        if not filtered_ads:
            return {'ad_id': self.fallback_ad, 'name': 'Default Promotion', 'score': 0}
            
        # Step 2: LinUCB Selection
        context = self._get_context_vector(age, gender, emotion)
        valid_arm_ids = [ad['ad_id'] for ad in filtered_ads]
        
        selected_id = self.linucb.select_arm(context, valid_arm_ids)
        
        # Find the selected ad object
        selected_ad = next((ad for ad in filtered_ads if ad['ad_id'] == selected_id), filtered_ads[0])
        
        # Update display history
        if track_id:
            self.display_history[track_id] = {
                'ad_id': selected_id,
                'context': context,
                'start_time': time.time()
            }
        
        return selected_ad
        
    def update_feedback(self, track_id: int, duration: float):
        """
        Update model with feedback (view duration)
        """
        if track_id in self.display_history:
            history = self.display_history[track_id]
            
            # Reward function: Logarithmic scaling of duration
            # 1s = 0, 5s ~ 1.6, 10s ~ 2.3
            reward = np.log(1 + duration)
            if duration > 10: reward += 1.0 # Bonus for very long views
            
            self.linucb.update(
                context=history['context'],
                arm_id=history['ad_id'],
                reward=reward
            )
            
            # Cleanup
            del self.display_history[track_id]
    
    def _get_context_vector(self, age: int, gender: str, emotion: str) -> np.ndarray:
        """Create feature vector from attributes"""
        # 1. Age (Normalized 0-1)
        age_feat = [min(age, 100) / 100.0]
        
        # 2. Gender (One-hot: Male, Female)
        gender_feat = [1, 0] if gender.lower() == 'male' else [0, 1]
        
        # 3. Emotion (One-hot)
        emotion_vec = [0] * 7
        idx = self.emotion_map.get(emotion.lower(), 2) # Default neutral
        emotion_vec[idx] = 1
        
        # 4. Time (Normalized hour 0-1)
        current_hour = datetime.now().hour / 24.0
        time_feat = [current_hour]
        
        return np.array(age_feat + gender_feat + emotion_vec + time_feat)
    
    def _filter_ads(self, age: int, gender: str, emotion: str) -> List[Dict]:
        """Filter ads based on age, gender, and emotion criteria"""
        filtered = []
        for ad in self.ad_rules:
            # Age filter
            if not (ad['target_age_min'] <= age <= ad['target_age_max']):
                continue
            # Gender filter
            if 'all' not in ad['target_gender'] and gender not in ad['target_gender']:
                continue
            # Emotion filter
            if 'all' not in ad['target_emotions'] and emotion not in ad['target_emotions']:
                continue
            filtered.append(ad)
        return filtered


