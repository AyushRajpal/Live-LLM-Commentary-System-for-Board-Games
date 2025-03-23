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
# from checkers_evaluator import CheckersAnalyzer
from config_util import get_config
from text2speech import CommandTTS
import utils

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from checkers_evaluator import CheckersAnalyzer
    from llm_integration import CommentaryGenerator

CONFIG = get_config()

class PieceType(Enum):
    EMPTY = 0
    WHITE = 1
    BLACK = 2
    WHITE_KING = 3
    BLACK_KING = 4

class CheckersGame:
    """Implementation of a checkers game with rules and move validation."""
    
    def __init__(self, board_size=8):
        self.board_size = board_size
        self.board = self._init_board()
        self.current_player = PieceType.BLACK  # Black moves first
        self.move_history = []
        self.available_moves = self._get_all_available_moves()
        
    def _init_board(self):
        """Initialize the checkers board with pieces in starting positions."""
        board = np.zeros((self.board_size, self.board_size), dtype=int)
        
        # Place white pieces
        for row in range(0, 3):
            for col in range(self.board_size):
                if (row + col) % 2 == 1:
                    board[row, col] = PieceType.WHITE.value
        
        # Place black pieces
        for row in range(5, 8):
            for col in range(self.board_size):
                if (row + col) % 2 == 1:
                    board[row, col] = PieceType.BLACK.value
        
        return board
    
    def _is_king_row(self, row:int, piece_type:PieceType):
        """Check if the piece has reached the king row."""
        if piece_type == PieceType.BLACK:
            return row == 0
        elif piece_type == PieceType.WHITE:
            return row == self.board_size - 1
        return False
    
    def _get_piece_moves(self, row:int, col:int):
        """Get all valid moves for a specific piece."""
        piece = PieceType(self.board[row, col])
        if piece == PieceType.EMPTY or (piece == PieceType.WHITE and self.current_player == PieceType.BLACK) or (piece == PieceType.BLACK and self.current_player == PieceType.WHITE):
            return []
        
        moves = []
        captures = []
        
        # Direction of movement (rows)
        directions = []
        if piece in [PieceType.BLACK, PieceType.BLACK_KING, PieceType.WHITE_KING]:
            directions.append(-1)  # Up
        if piece in [PieceType.WHITE, PieceType.BLACK_KING, PieceType.WHITE_KING]:
            directions.append(1)   # Down
            
        for d_row in directions:
            for d_col in [-1, 1]:  # Diagonal left and right
                new_row, new_col = row + d_row, col + d_col
                
                # Check if the move is within the board
                if 0 <= new_row < self.board_size and 0 <= new_col < self.board_size:
                    # Regular move (no capture)
                    if self.board[new_row, new_col] == PieceType.EMPTY.value:
                        moves.append(((row, col), (new_row, new_col)))
                    
                    # Capture move
                    elif ((self.board[new_row, new_col] in [PieceType.WHITE.value, PieceType.WHITE_KING.value] and 
                           self.current_player == PieceType.BLACK) or 
                          (self.board[new_row, new_col] in [PieceType.BLACK.value, PieceType.BLACK_KING.value] and 
                           self.current_player == PieceType.WHITE)):
                        
                        jump_row, jump_col = new_row + d_row, new_col + d_col
                        if (0 <= jump_row < self.board_size and 
                            0 <= jump_col < self.board_size and 
                            self.board[jump_row, jump_col] == PieceType.EMPTY.value):
                            captures.append(((row, col), (jump_row, jump_col)))
        
        # If captures are available, only return captures (forced capture rule)
        if captures:
            return captures
        return moves
    
    def _get_all_available_moves(self):
        """Get all available moves for the current player."""
        all_moves = []
        all_captures = []
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = PieceType(self.board[row, col])
                if ((piece == PieceType.BLACK or piece == PieceType.BLACK_KING) and self.current_player == PieceType.BLACK) or \
                   ((piece == PieceType.WHITE or piece == PieceType.WHITE_KING) and self.current_player == PieceType.WHITE):
                    moves = self._get_piece_moves(row, col)
                    # Check if any moves are captures
                    for move in moves:
                        start, end = move
                        if abs(end[0] - start[0]) > 1 or abs(end[1] - start[1]) > 1:
                            all_captures.append(move)
                        else:
                            all_moves.append(move)
        
        # If there are captures available, return only captures
        if all_captures:
            return all_captures
        return all_moves
    
    def make_move(self, start_pos, end_pos):
        """Make a move on the board."""
        start_row, start_col = start_pos
        end_row, end_col = end_pos
        
        # Validate move
        move = (start_pos, end_pos)
        if move not in self.available_moves:
            return False, "Invalid move"
        
        # Check if it's a capture move
        is_capture = abs(end_row - start_row) > 1
        
        # Get the piece and update the board
        piece = PieceType(self.board[start_row, start_col])
        self.board[start_row, start_col] = PieceType.EMPTY.value
        
        # Handle promotion to king
        if self._is_king_row(end_row, piece):
            if piece == PieceType.BLACK:
                self.board[end_row, end_col] = PieceType.BLACK_KING.value
            elif piece == PieceType.WHITE:
                self.board[end_row, end_col] = PieceType.WHITE_KING.value
        else:
            self.board[end_row, end_col] = piece.value
        
        # Handle capture by removing the captured piece
        if is_capture:
            captured_row = (start_row + end_row) // 2
            captured_col = (start_col + end_col) // 2
            self.board[captured_row, captured_col] = PieceType.EMPTY.value
        
        # Record the move
        self.move_history.append({
            "player": self.current_player.name,
            "start": start_pos,
            "end": end_pos,
            "is_capture": is_capture,
            "piece_type": piece.name
        })
        
        # Check for additional captures
        additional_captures = []
        if is_capture:
            # Check if the same piece can capture more
            additional_captures = self._get_piece_moves(end_row, end_col)
            if additional_captures:
                # Only keep the additional captures, not regular moves
                additional_captures = [move for move in additional_captures 
                                     if abs(move[1][0] - move[0][0]) > 1]
        
        if not additional_captures:
            # Switch player if no additional captures
            self.current_player = PieceType.WHITE if self.current_player == PieceType.BLACK else PieceType.BLACK
            self.available_moves = self._get_all_available_moves()
        else:
            # Only allow the same piece to move for additional captures
            self.available_moves = additional_captures
        
        return True, None
    
    def get_winner(self):
        """Check if there's a winner."""
        if not self.available_moves:
            return PieceType.WHITE if self.current_player == PieceType.BLACK else PieceType.BLACK
        
        # Check if any player has no pieces left
        white_pieces = np.count_nonzero((self.board == PieceType.WHITE.value) | 
                                         (self.board == PieceType.WHITE_KING.value))
        black_pieces = np.count_nonzero((self.board == PieceType.BLACK.value) | 
                                         (self.board == PieceType.BLACK_KING.value))
        
        if white_pieces == 0:
            return PieceType.BLACK
        elif black_pieces == 0:
            return PieceType.WHITE
        
        return None
    
    def get_board_state(self):
        """Return the current board state as a 2D array."""
        return self.board.copy()



class CheckersUI:
    """User interface for the checkers game and commentary."""
    
    def __init__(self, game:CheckersGame, analyzer:'CheckersAnalyzer', commentator:'CommentaryGenerator'):
        logging.info("Initializing CheckersUI")
        self.game = game
        self.analyzer = analyzer
        self.commentator = commentator
        
        # Initialize pygame
        pygame.init()
        self.window_width = CONFIG["window_width"]
        self.window_height = CONFIG["window_height"]
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Checkers with Live Commentary")
        
        # Calculate square size
        self.board_size = CONFIG["board_size"]
        self.board_padding = 20
        self.board_width = min(self.window_width - 300, self.window_height - 2 * self.board_padding)
        self.square_size = self.board_width // self.board_size
        
        # Piece selection
        self.selected_piece = None
        self.valid_moves = []
        
        # Colors
        self.colors = {
            "light_square": (240, 217, 181),   # Light brown
            "dark_square": (181, 136, 99),     # Dark brown
            "highlight": (124, 252, 0),        # Green highlight
            "white_piece": (255, 255, 255),    # White
            "black_piece": (0, 0, 0),          # Black
            "text_bg": (50, 50, 50),           # Dark gray
            "text": (255, 255, 255),           # White text
            "valid_move": (255, 255, 0, 128)   # Semi-transparent yellow
        }
        
        # Font
        self.font = pygame.font.SysFont("Arial", 16)
        self.title_font = pygame.font.SysFont("Arial", 20, bold=True)
        
        # Commentary
        self.current_commentary = "Game started. Waiting for the first move..."
        
        # Text-to-speech
        if CONFIG["text_to_speech"]:
            self.tts = CommandTTS()
        else:
            self.tts = None
            
        logging.info("CheckersUI initialized successfully")
    
    def handle_events(self):
        """Handle pygame events."""
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Get board coordinates
                    mouse_pos = pygame.mouse.get_pos()
                    
                    # Check if click is within the board area
                    board_left = self.board_padding
                    board_top = (self.window_height - self.board_width) // 2
                    
                    if (board_left <= mouse_pos[0] <= board_left + self.board_width and
                        board_top <= mouse_pos[1] <= board_top + self.board_width):
                        
                        # Convert to board coordinates
                        col = (mouse_pos[0] - board_left) // self.square_size
                        row = (mouse_pos[1] - board_top) // self.square_size
                        
                        self.handle_board_click(row, col)
            
            return True
        except Exception as e:
            logging.error(f"Error in handle_events: {str(e)}")
            logging.error(traceback.format_exc())
            return True  # Keep the game running even if there's an error
    
    def handle_board_click(self, row, col):
        """Handle a click on the board."""
        try:
            # If no piece is selected
            if self.selected_piece is None:
                piece = PieceType(self.game.board[row, col])
                
                # Check if the piece belongs to the current player
                if ((piece == PieceType.BLACK or piece == PieceType.BLACK_KING) and 
                    self.game.current_player == PieceType.BLACK) or \
                   ((piece == PieceType.WHITE or piece == PieceType.WHITE_KING) and 
                    self.game.current_player == PieceType.WHITE):
                    
                    # Select the piece
                    self.selected_piece = (row, col)
                    
                    # Get valid moves for this piece
                    self.valid_moves = [move for move in self.game.available_moves 
                                      if move[0] == (row, col)]
            
            # If a piece is already selected
            else:
                # Check if the click is on a valid destination
                for move in self.valid_moves:
                    if move[1] == (row, col):
                        # Make the move
                        success, error = self.game.make_move(self.selected_piece, (row, col))
                        
                        if success:
                            # Generate new commentary
                            analysis = self.analyzer.analyze_game_state()
                            self.current_commentary = self.commentator.generate_commentary(self.game, analysis)
                            
                            # Speak the commentary
                            if self.tts:
                                self.tts.speak(self.current_commentary)
                        
                        # Reset selection
                        self.selected_piece = None
                        self.valid_moves = []
                        return
                
                # If clicked on another one of player's pieces, select that piece instead
                piece = PieceType(self.game.board[row, col])
                if ((piece == PieceType.BLACK or piece == PieceType.BLACK_KING) and 
                    self.game.current_player == PieceType.BLACK) or \
                   ((piece == PieceType.WHITE or piece == PieceType.WHITE_KING) and 
                    self.game.current_player == PieceType.WHITE):
                    
                    self.selected_piece = (row, col)
                    self.valid_moves = [move for move in self.game.available_moves 
                                      if move[0] == (row, col)]
                else:
                    # Deselect the piece
                    self.selected_piece = None
                    self.valid_moves = []
        except Exception as e:
            logging.error(f"Error in handle_board_click: {str(e)}")
            logging.error(traceback.format_exc())
            # Reset selection to avoid getting stuck
            self.selected_piece = None
            self.valid_moves = []
    
    def draw_board(self):
        """Draw the checkers board."""
        try:
            # Draw background
            self.screen.fill((30, 30, 30))
            
            # Calculate board position
            board_left = self.board_padding
            board_top = (self.window_height - self.board_width) // 2
            
            # Draw squares
            for row in range(self.board_size):
                for col in range(self.board_size):
                    square_color = self.colors["light_square"] if (row + col) % 2 == 0 else self.colors["dark_square"]
                    
                    # Highlight selected piece
                    if self.selected_piece and self.selected_piece == (row, col):
                        square_color = self.colors["highlight"]
                    
                    # Draw square
                    pygame.draw.rect(
                        self.screen,
                        square_color,
                        (
                            board_left + col * self.square_size,
                            board_top + row * self.square_size,
                            self.square_size,
                            self.square_size
                        )
                    )
                    
                    # Highlight valid moves
                    if self.selected_piece:
                        for move in self.valid_moves:
                            if move[1] == (row, col):
                                # Create a transparent surface
                                s = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
                                s.fill(self.colors["valid_move"])
                                self.screen.blit(
                                    s,
                                    (
                                        board_left + col * self.square_size,
                                        board_top + row * self.square_size
                                    )
                                )
            
            # Draw pieces
            for row in range(self.board_size):
                for col in range(self.board_size):
                    piece = PieceType(self.game.board[row, col])
                    
                    if piece != PieceType.EMPTY:
                        # Determine piece color
                        if piece == PieceType.WHITE or piece == PieceType.WHITE_KING:
                            piece_color = self.colors["white_piece"]
                        else:
                            piece_color = self.colors["black_piece"]
                        
                        # Calculate center of square
                        center_x = board_left + col * self.square_size + self.square_size // 2
                        center_y = board_top + row * self.square_size + self.square_size // 2
                        
                        # Draw piece
                        radius = self.square_size // 2 - 5
                        pygame.draw.circle(self.screen, piece_color, (center_x, center_y), radius)
                        
                        # Draw border for contrast
                        pygame.draw.circle(self.screen, self.colors["dark_square"] 
                                          if piece_color == self.colors["white_piece"] 
                                          else self.colors["light_square"], 
                                          (center_x, center_y), radius, 2)
                        
                        # Draw crown for kings
                        if piece == PieceType.WHITE_KING or piece == PieceType.BLACK_KING:
                            crown_color = self.colors["dark_square"] if piece_color == self.colors["white_piece"] else self.colors["white_piece"]
                            pygame.draw.circle(self.screen, crown_color, (center_x, center_y), radius // 2)
            
            # Draw coordinate labels
            label_font = pygame.font.SysFont("Arial", 12)
            for i in range(self.board_size):
                # Column labels (letters A-H)
                col_label = chr(65 + i)
                text_surface = label_font.render(col_label, True, self.colors["text"])
                self.screen.blit(text_surface, (board_left + i * self.square_size + self.square_size // 2 - 5, 
                                               board_top - 20))
                
                # Row labels (numbers 1-8)
                row_label = str(i + 1)
                text_surface = label_font.render(row_label, True, self.colors["text"])
                self.screen.blit(text_surface, (board_left - 20, 
                                               board_top + i * self.square_size + self.square_size // 2 - 5))
        except Exception as e:
            logging.error(f"Error in draw_board: {str(e)}")
            logging.error(traceback.format_exc())
    
    def draw_game_info(self):
        """Draw game information and commentary."""
        try:
            info_left = self.board_padding * 2 + self.board_width
            info_width = self.window_width - info_left - self.board_padding
            
            # Title
            title_surface = self.title_font.render("Live Commentary", True, self.colors["text"])
            self.screen.blit(title_surface, (info_left, 30))
            
            # Current player
            player_text = f"Current Player: {self.game.current_player.name}"
            player_surface = self.font.render(player_text, True, self.colors["text"])
            self.screen.blit(player_surface, (info_left, 70))
            
            # Piece counts
            board = self.game.get_board_state()
            white_pieces = np.count_nonzero((board == PieceType.WHITE.value) | 
                                            (board == PieceType.WHITE_KING.value))
            black_pieces = np.count_nonzero((board == PieceType.BLACK.value) | 
                                            (board == PieceType.BLACK_KING.value))
            
            pieces_text = f"Black: {black_pieces} pieces | White: {white_pieces} pieces"
            pieces_surface = self.font.render(pieces_text, True, self.colors["text"])
            self.screen.blit(pieces_surface, (info_left, 100))
            
            # Move count
            moves_text = f"Moves: {len(self.game.move_history)}"
            moves_surface = self.font.render(moves_text, True, self.colors["text"])
            self.screen.blit(moves_surface, (info_left, 130))
            
            # Commentary box
            pygame.draw.rect(
                self.screen,
                self.colors["text_bg"],
                (info_left, 170, info_width, 200),
                border_radius=5
            )
            
            # Render commentary with word wrapping
            words = self.current_commentary.split(' ')
            lines = []
            current_line = []
            
            for word in words:
                test_line = ' '.join(current_line + [word])
                test_surface = self.font.render(test_line, True, self.colors["text"])
                
                if test_surface.get_width() <= info_width - 20:
                    current_line.append(word)
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
            
            if current_line:
                lines.append(' '.join(current_line))
            
            for i, line in enumerate(lines):
                line_surface = self.font.render(line, True, self.colors["text"])
                self.screen.blit(line_surface, (info_left + 10, 180 + i * 25))
            
            # Commentary style
            style_text = f"Commentary Style: {CONFIG['commentary_style'].title()}"
            style_surface = self.font.render(style_text, True, self.colors["text"])
            self.screen.blit(style_surface, (info_left, 380))
            
            # Text-to-speech status
            tts_text = f"Text-to-Speech: {'On' if CONFIG['text_to_speech'] else 'Off'}"
            tts_surface = self.font.render(tts_text, True, self.colors["text"])
            self.screen.blit(tts_surface, (info_left, 410))
            
            # LLM Provider
            provider_text = f"LLM Provider: {CONFIG['llm_provider'].title()}"
            provider_surface = self.font.render(provider_text, True, self.colors["text"])
            self.screen.blit(provider_surface, (info_left, 440))
            
            # Instructions
            instructions_text = "Click on a piece to select, then click destination"
            instructions_surface = self.font.render(instructions_text, True, self.colors["text"])
            self.screen.blit(instructions_surface, (info_left, 470))
        except Exception as e:
            logging.error(f"Error in draw_game_info: {str(e)}")
            logging.error(traceback.format_exc())
    
    def run(self):
        """Main game loop."""
        try:
            logging.info("Starting game loop")
            running = True
            clock = pygame.time.Clock()
            
            # Initial analysis and commentary
            analysis = self.analyzer.analyze_game_state()
            self.current_commentary = self.commentator.generate_commentary(self.game, analysis)
            
            # Text-to-speech for initial commentary
            if self.tts:
                self.tts.speak(self.current_commentary)
            
            while running:
                # Handle events
                running = self.handle_events()
                
                # Draw the game
                self.draw_board()
                self.draw_game_info()
                
                # Check for game over
                winner = self.game.get_winner()
                if winner:
                    # Display winner
                    win_text = f"{winner.name} wins!"
                    win_surface = pygame.font.SysFont("Arial", 32, bold=True).render(win_text, True, (255, 215, 0))
                    win_rect = win_surface.get_rect(center=(self.window_width // 2, self.window_height // 2))
                    
                    # Draw background
                    bg_rect = win_rect.inflate(20, 20)
                    pygame.draw.rect(self.screen, (0, 0, 0, 200), bg_rect, border_radius=10)
                    
                    # Draw text
                    self.screen.blit(win_surface, win_rect)
                
                # Update the display
                pygame.display.flip()
                
                # Cap the frame rate
                clock.tick(30)
            
            # Cleanup
            if self.tts:
                self.tts.stop()
            pygame.quit()
            logging.info("Game ended normally")
        except Exception as e:
            logging.error(f"Error in game loop: {str(e)}")
            logging.error(traceback.format_exc())
            # Attempt to clean up
            if self.tts:
                self.tts.stop()
            pygame.quit()
