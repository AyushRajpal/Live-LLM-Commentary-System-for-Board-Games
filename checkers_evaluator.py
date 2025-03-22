# library imports
import numpy as np
import requests
import pygame

# python imports
import time
import json
import os
from enum import Enum
import threading
import logging
import traceback
import sys
import subprocess

# local imports
from checkers_game import CheckersGame, PieceType
import utils

class CheckersAnalyzer:
    """Analyzes the current board state and suggests optimal moves."""
    
    def __init__(self, game):
        self.game = game
        
    def evaluate_board(self):
        """Evaluate the current board state with a simple heuristic."""
        board = self.game.get_board_state()
        
        # Simple piece count heuristic
        white_value = np.count_nonzero(board == PieceType.WHITE.value) + 1.5 * np.count_nonzero(board == PieceType.WHITE_KING.value)
        black_value = np.count_nonzero(board == PieceType.BLACK.value) + 1.5 * np.count_nonzero(board == PieceType.BLACK_KING.value)
        
        if self.game.current_player == PieceType.BLACK:
            return black_value - white_value
        else:
            return white_value - black_value
    
    def minimax(self, depth, alpha, beta, maximizing_player):
        """Minimax algorithm with alpha-beta pruning."""
        if depth == 0 or self.game.get_winner() is not None:
            return self.evaluate_board(), None
        
        best_move = None
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in self.game.available_moves:
                # Make a copy of the game to simulate the move
                game_copy = CheckersGame(self.game.board_size)
                game_copy.board = self.game.board.copy()
                game_copy.current_player = self.game.current_player
                game_copy.available_moves = self.game.available_moves.copy()
                
                # Make the move
                game_copy.make_move(move[0], move[1])
                
                # Recursively evaluate
                analyzer = CheckersAnalyzer(game_copy)
                eval_score, _ = analyzer.minimax(depth - 1, alpha, beta, False)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
                
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in self.game.available_moves:
                # Make a copy of the game to simulate the move
                game_copy = CheckersGame(self.game.board_size)
                game_copy.board = self.game.board.copy()
                game_copy.current_player = self.game.current_player
                game_copy.available_moves = self.game.available_moves.copy()
                
                # Make the move
                game_copy.make_move(move[0], move[1])
                
                # Recursively evaluate
                analyzer = CheckersAnalyzer(game_copy)
                eval_score, _ = analyzer.minimax(depth - 1, alpha, beta, True)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
                
            return min_eval, best_move
    
    def get_best_move(self, depth=3):
        """Get the best move for the current player using minimax algorithm."""
        try:
            _, best_move = self.minimax(depth, float('-inf'), float('inf'), True)
            return best_move
        except Exception as e:
            logging.error(f"Error in get_best_move: {str(e)}")
            logging.error(traceback.format_exc())
            return None
    
    def analyze_game_state(self):
        """Analyze the current game state and return observations."""
        try:
            board = self.game.get_board_state()
            current_player = self.game.current_player
            
            # Count pieces
            white_pieces = np.count_nonzero((board == PieceType.WHITE.value) | 
                                             (board == PieceType.WHITE_KING.value))
            black_pieces = np.count_nonzero((board == PieceType.BLACK.value) | 
                                             (board == PieceType.BLACK_KING.value))
            
            # Get best move
            best_move = self.get_best_move(depth=1)  # Reduced depth for performance
            
            # Material advantage
            material_diff = black_pieces - white_pieces
            if material_diff > 0:
                material_advantage = "Black has a material advantage"
            elif material_diff < 0:
                material_advantage = "White has a material advantage"
            else:
                material_advantage = "Material is even"
            
            # Analyze the best move
            move_analysis = "No moves available"
            if best_move:
                # Convert to chess notation for display
                start_notation = utils.coord_to_notation(best_move[0][0], best_move[0][1])
                end_notation = utils.coord_to_notation(best_move[1][0], best_move[1][1])
                
                is_capture = abs(best_move[1][0] - best_move[0][0]) > 1
                move_type = "capture" if is_capture else "move"
                move_analysis = f"Best {move_type} is from {start_notation} to {end_notation}"
            
            # Game phase estimation
            total_pieces = white_pieces + black_pieces
            if total_pieces > 18:
                phase = "opening"
            elif total_pieces > 10:
                phase = "middle game"
            else:
                phase = "endgame"
            
            # Check for potential king promotions
            potential_kings = []
            for move in self.game.available_moves:
                start, end = move
                piece = PieceType(board[start[0], start[1]])
                if ((piece == PieceType.BLACK and end[0] == 0) or 
                    (piece == PieceType.WHITE and end[0] == 7)):
                    potential_kings.append(move)
            
            king_promotion = "No potential king promotions"
            if potential_kings:
                king_promotion = f"Potential king promotion(s) available: {len(potential_kings)}"
            
            analysis = {
                "current_player": current_player.name,
                "material_advantage": material_advantage,
                "material_difference": abs(material_diff),
                "best_move": best_move,
                "move_analysis": move_analysis,
                "game_phase": phase,
                "king_promotion": king_promotion,
                "white_pieces": white_pieces,
                "black_pieces": black_pieces,
                "available_moves": len(self.game.available_moves)
            }
            
            return analysis
        except Exception as e:
            logging.error(f"Error in analyze_game_state: {str(e)}")
            logging.error(traceback.format_exc())
            # Return a safe default analysis
            return {
                "current_player": self.game.current_player.name,
                "material_advantage": "Analysis unavailable",
                "material_difference": 0,
                "best_move": None,
                "move_analysis": "Analysis unavailable",
                "game_phase": "opening",
                "king_promotion": "Analysis unavailable",
                "white_pieces": 12,
                "black_pieces": 12,
                "available_moves": len(self.game.available_moves)
            }
