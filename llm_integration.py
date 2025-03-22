
import time
import json
import logging
import traceback
import requests
from config import get_config

CONFIG = get_config()


class CommentaryGenerator:
    """Generates game commentary using LLM."""
    
    def __init__(self, style="educational"):
        self.style = style  # educational, casual, expert
        self.commentary_history = []
    
    def _build_gemini_prompt(self, game_state, analysis):
        """Build a prompt for the Gemini API based on the game state and analysis."""
        prompt = f"""
        You are an expert checkers commentator with a {self.style} style. Analyze this game state and provide insightful commentary.
        
        Game State:
        {game_state}
        
        Analysis:
        - Current player: {analysis['current_player']}
        - Material: {analysis['material_advantage']} ({abs(analysis['material_difference'])} pieces)
        - Game phase: {analysis['game_phase']}
        - Available moves: {analysis['available_moves']}
        - Best move analysis: {analysis['move_analysis']}
        - King promotion opportunities: {analysis['king_promotion']}
        
        Based on this information, provide a brief, natural-sounding commentary that explains the current situation, 
        evaluates the position, and discusses potential strategy. Keep your response between 2-4 sentences.
        """
        return prompt
    
    def _call_gemini_api(self, prompt):
        """Call the Gemini API to generate commentary."""
        if not CONFIG["api_key"]:
            return "API key not configured. This is a placeholder commentary."
        
        try:
            # Basic API call to Gemini - implement actual API call
            url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": CONFIG["api_key"]
            }
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generation_config": {
                    "temperature": 0.7,
                    "maxOutputTokens": 200
                }
            }
            
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                response_json = response.json()
                if "candidates" in response_json and len(response_json["candidates"]) > 0:
                    return response_json["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    return "Error parsing Gemini API response."
            else:
                return f"API error: {response.status_code}"
            
        except Exception as e:
            logging.error(f"Error calling Gemini API: {str(e)}")
            return f"Error calling Gemini API: {str(e)}"
    
    def _generate_mock_commentary(self, game_state, analysis):
        """Generate mock commentary for testing without API calls."""
        player = analysis['current_player']
        material = analysis['material_advantage']
        phase = analysis['game_phase']
        
        commentary_templates = [
            f"In this {phase}, {player} is preparing to make a move. {material}.",
            f"{player} needs to carefully consider their options. {analysis['move_analysis']}.",
            f"We're seeing a typical {phase} position here with {analysis['black_pieces']} black pieces and {analysis['white_pieces']} white pieces.",
            f"The tension is building as {player} contemplates their next move. {analysis['king_promotion']}.",
            f"This is a critical moment in the {phase}. {player} has {analysis['available_moves']} possible moves to consider.",
            f"A fascinating position has developed in this {phase}. {material} by {analysis['material_difference']} pieces.",
            f"As we progress through this {phase}, {player} must decide whether to be aggressive or defensive.",
            f"The board position shows classic {phase} characteristics. {player} needs to think strategically here.",
        ]
        
        import random
        return random.choice(commentary_templates)
    
    def generate_commentary(self, game, analysis):
        """Generate commentary based on the current game state."""
        try:
            game_state = game.get_game_state_description()
            
            # Choose the appropriate LLM provider
            if CONFIG["llm_provider"] == "gemini":
                prompt = self._build_gemini_prompt(game_state, analysis)
                commentary = self._call_gemini_api(prompt)
            elif CONFIG["llm_provider"] == "openai":
                # Implement OpenAI integration here
                commentary = "OpenAI integration not implemented yet."
            else:
                # Mock commentary for testing
                commentary = self._generate_mock_commentary(game_state, analysis)
            
            # Add to history
            self.commentary_history.append({
                "timestamp": time.time(),
                "commentary": commentary
            })
            
            return commentary
        except Exception as e:
            logging.error(f"Error generating commentary: {str(e)}")
            logging.error(traceback.format_exc())
            return "The game continues with both players carefully considering their options."

