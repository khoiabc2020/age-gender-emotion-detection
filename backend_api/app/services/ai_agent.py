"""
AI Agent Service - Integration with Google AI and ChatGPT
Giai đoạn 6: Generative AI Analyst
"""

import os
from typing import Optional, Dict, List
from enum import Enum
import json
from datetime import datetime

try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class AIProvider(str, Enum):
    """AI Provider options"""
    GOOGLE_AI = "google_ai"
    CHATGPT = "chatgpt"
    BOTH = "both"  # Use both and combine results

class AIAgent:
    """
    AI Agent for analyzing retail analytics data
    Supports both Google AI (Gemini) and ChatGPT
    """
    
    def __init__(
        self,
        google_ai_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        provider: AIProvider = AIProvider.GOOGLE_AI
    ):
        """
        Initialize AI Agent
        
        Args:
            google_ai_api_key: Google AI API key
            openai_api_key: OpenAI API key
            provider: AI provider to use
        """
        self.provider = provider
        self.google_ai_key = google_ai_api_key or os.getenv("GOOGLE_AI_API_KEY")
        self.openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize Google AI
        if GOOGLE_AI_AVAILABLE and self.google_ai_key:
            genai.configure(api_key=self.google_ai_key)
            self.google_model = genai.GenerativeModel('gemini-pro')
        else:
            self.google_model = None
        
        # Initialize OpenAI
        if OPENAI_AVAILABLE and self.openai_key:
            openai.api_key = self.openai_key
            self.openai_client = openai
        else:
            self.openai_client = None
    
    def analyze_analytics(
        self,
        stats: Dict,
        age_by_hour: List[Dict],
        emotion_distribution: List[Dict],
        gender_distribution: Dict,
        ad_performance: List[Dict]
    ) -> Dict:
        """
        Analyze retail analytics data using AI
        
        Args:
            stats: Overall statistics
            age_by_hour: Age distribution by hour
            emotion_distribution: Emotion distribution
            gender_distribution: Gender distribution
            ad_performance: Advertisement performance
            
        Returns:
            Analysis results with insights and recommendations
        """
        # Prepare data summary
        data_summary = self._prepare_data_summary(
            stats, age_by_hour, emotion_distribution, gender_distribution, ad_performance
        )
        
        # Generate prompt
        prompt = self._create_analysis_prompt(data_summary)
        
        # Get AI response
        if self.provider == AIProvider.GOOGLE_AI or self.provider == AIProvider.BOTH:
            google_response = self._query_google_ai(prompt)
        else:
            google_response = None
        
        if self.provider == AIProvider.CHATGPT or self.provider == AIProvider.BOTH:
            chatgpt_response = self._query_chatgpt(prompt)
        else:
            chatgpt_response = None
        
        # Combine results if using both
        if self.provider == AIProvider.BOTH and google_response and chatgpt_response:
            return self._combine_responses(google_response, chatgpt_response)
        elif google_response:
            return google_response
        elif chatgpt_response:
            return chatgpt_response
        else:
            return {
                "error": "No AI provider available. Please configure API keys.",
                "insights": [],
                "recommendations": []
            }
    
    def chat_with_data(
        self,
        question: str,
        context_data: Dict
    ) -> str:
        """
        Chat with analytics data
        
        Args:
            question: User question
            context_data: Analytics data context
            
        Returns:
            AI response
        """
        prompt = f"""
        You are an AI analyst for a retail analytics system. 
        Answer the following question based on the provided data.
        
        Question: {question}
        
        Data Context:
        {json.dumps(context_data, indent=2)}
        
        Provide a clear, concise answer with insights.
        """
        
        if self.provider == AIProvider.GOOGLE_AI or self.provider == AIProvider.BOTH:
            if self.google_model:
                response = self.google_model.generate_content(prompt)
                return response.text
        elif self.provider == AIProvider.CHATGPT:
            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an AI analyst for retail analytics."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content
        
        return "AI service not available. Please configure API keys."
    
    def generate_report(
        self,
        stats: Dict,
        time_range: str = "24 hours"
    ) -> str:
        """
        Generate automated analytics report
        
        Args:
            stats: Statistics data
            time_range: Time range for report
            
        Returns:
            Generated report text
        """
        prompt = f"""
        Generate a comprehensive analytics report for a retail system.
        
        Time Range: {time_range}
        
        Statistics:
        - Total Interactions: {stats.get('total_interactions', 0)}
        - Unique Customers: {stats.get('unique_customers', 0)}
        - Average Age: {stats.get('avg_age', 0):.1f} years
        - Gender Distribution: {json.dumps(stats.get('gender_distribution', {}))}
        - Emotion Distribution: {json.dumps(stats.get('emotion_distribution', {}))}
        
        Create a professional report with:
        1. Executive Summary
        2. Key Metrics
        3. Insights
        4. Recommendations
        5. Next Steps
        
        Format as markdown.
        """
        
        if self.provider == AIProvider.GOOGLE_AI or self.provider == AIProvider.BOTH:
            if self.google_model:
                response = self.google_model.generate_content(prompt)
                return response.text
        elif self.provider == AIProvider.CHATGPT:
            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a professional business analyst."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content
        
        return "AI service not available. Please configure API keys."
    
    def _prepare_data_summary(
        self,
        stats: Dict,
        age_by_hour: List[Dict],
        emotion_distribution: List[Dict],
        gender_distribution: Dict,
        ad_performance: List[Dict]
    ) -> str:
        """Prepare data summary for AI analysis"""
        summary = {
            "overall_stats": stats,
            "age_by_hour": age_by_hour,
            "emotion_distribution": emotion_distribution,
            "gender_distribution": gender_distribution,
            "top_ads": ad_performance[:5] if ad_performance else []
        }
        return json.dumps(summary, indent=2)
    
    def _create_analysis_prompt(self, data_summary: str) -> str:
        """Create analysis prompt for AI"""
        return f"""
        You are an expert retail analytics AI. Analyze the following data and provide:
        
        1. Key Insights (3-5 bullet points)
        2. Trends and Patterns
        3. Recommendations for improvement
        4. Action Items
        
        Data:
        {data_summary}
        
        Provide your analysis in JSON format:
        {{
            "insights": ["insight1", "insight2", ...],
            "trends": ["trend1", "trend2", ...],
            "recommendations": ["rec1", "rec2", ...],
            "action_items": ["action1", "action2", ...]
        }}
        """
    
    def _query_google_ai(self, prompt: str) -> Dict:
        """Query Google AI (Gemini)"""
        if not self.google_model:
            return {"error": "Google AI not configured"}
        
        try:
            response = self.google_model.generate_content(prompt)
            text = response.text
            
            # Try to parse JSON from response
            try:
                # Extract JSON from markdown code blocks if present
                if "```json" in text:
                    json_start = text.find("```json") + 7
                    json_end = text.find("```", json_start)
                    text = text[json_start:json_end].strip()
                elif "```" in text:
                    json_start = text.find("```") + 3
                    json_end = text.find("```", json_start)
                    text = text[json_start:json_end].strip()
                
                result = json.loads(text)
            except:
                # If not JSON, create structured response
                result = {
                    "insights": [text],
                    "trends": [],
                    "recommendations": [],
                    "action_items": [],
                    "raw_response": text
                }
            
            return {
                "provider": "google_ai",
                "timestamp": datetime.utcnow().isoformat(),
                **result
            }
        except Exception as e:
            return {
                "error": f"Google AI error: {str(e)}",
                "insights": [],
                "recommendations": []
            }
    
    def _query_chatgpt(self, prompt: str) -> Dict:
        """Query ChatGPT"""
        if not self.openai_client:
            return {"error": "ChatGPT not configured"}
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert retail analytics AI. Always respond in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            text = response.choices[0].message.content
            
            # Try to parse JSON
            try:
                if "```json" in text:
                    json_start = text.find("```json") + 7
                    json_end = text.find("```", json_start)
                    text = text[json_start:json_end].strip()
                elif "```" in text:
                    json_start = text.find("```") + 3
                    json_end = text.find("```", json_start)
                    text = text[json_start:json_end].strip()
                
                result = json.loads(text)
            except:
                result = {
                    "insights": [text],
                    "trends": [],
                    "recommendations": [],
                    "action_items": [],
                    "raw_response": text
                }
            
            return {
                "provider": "chatgpt",
                "timestamp": datetime.utcnow().isoformat(),
                **result
            }
        except Exception as e:
            return {
                "error": f"ChatGPT error: {str(e)}",
                "insights": [],
                "recommendations": []
            }
    
    def _combine_responses(self, google_response: Dict, chatgpt_response: Dict) -> Dict:
        """Combine responses from both providers"""
        return {
            "provider": "both",
            "timestamp": datetime.utcnow().isoformat(),
            "google_ai": google_response,
            "chatgpt": chatgpt_response,
            "combined_insights": list(set(
                google_response.get("insights", []) + 
                chatgpt_response.get("insights", [])
            )),
            "combined_recommendations": list(set(
                google_response.get("recommendations", []) + 
                chatgpt_response.get("recommendations", [])
            ))
        }

