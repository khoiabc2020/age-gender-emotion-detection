import requests
import random
import os
import threading
from typing import Optional

class GenerativeAds:
    """
    Generates dynamic ad copy/slogans using LLM (Gemini/Phi-3) or Templates.
    Designed to be non-blocking.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.cache = {}
        # Templates for fallback
        self.templates = {
            'happy': [
                "Enjoy your moment!",
                "Smile more with us!",
                "Great vibes, great offers!"
            ],
            'sad': [
                "Cheer up with a treat!",
                "You deserve a break.",
                "Small joy for you."
            ],
            'neutral': [
                "Check this out!",
                "Best choice for you.",
                "Discover something new."
            ],
            'surprise': [
                "Wow! unexpected deal!",
                "Surprise yourself!",
                "Amazing offer inside."
            ]
        }
        
    def generate_slogan(self, ad_name: str, emotion: str, age: int, gender: str) -> str:
        """
        Get a slogan. Tries API first, falls back to templates.
        This method should be called asynchronously if using API.
        """
        cache_key = f"{ad_name}_{emotion}_{gender}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # Try API if key exists
        if self.api_key:
            try:
                slogan = self._call_gemini(ad_name, emotion, age, gender)
                if slogan:
                    self.cache[cache_key] = slogan
                    return slogan
            except Exception as e:
                print(f"GenAI Error: {e}")
                
        # Fallback
        return self._get_template(emotion)
        
    def _get_template(self, emotion: str) -> str:
        options = self.templates.get(emotion, self.templates['neutral'])
        return random.choice(options)
        
    def _call_gemini(self, ad_name: str, emotion: str, age: int, gender: str) -> Optional[str]:
        """Call Google Gemini API"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.api_key}"
        
        prompt = (
            f"Write a very short (max 6 words), catchy slogan for an ad about '{ad_name}'. "
            f"Target audience: {age} year old {gender}, currently feeling {emotion}. "
            f"Do not use quotes. Just the text."
        )
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        response = requests.post(url, json=payload, timeout=2.0)
        
        if response.status_code == 200:
            data = response.json()
            try:
                text = data['candidates'][0]['content']['parts'][0]['text']
                return text.strip()
            except:
                return None
        return None
