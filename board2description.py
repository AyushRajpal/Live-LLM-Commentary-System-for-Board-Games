# this is a seperate file as it is a large block of code
# also to make code merges less conflict prone
from checkers_game import PieceType, CheckersGame
import numpy as np
import utils


def get_game_state_description(game_instance: CheckersGame) -> str:
    """Generate a text description of the current game state for the LLM."""
    board_state = game_instance.get_board_state()
    
    # Count pieces
    white_pieces = np.count_nonzero((board_state == PieceType.WHITE.value) | 
                                     (board_state == PieceType.WHITE_KING.value))
    black_pieces = np.count_nonzero((board_state == PieceType.BLACK.value) | 
                                     (board_state == PieceType.BLACK_KING.value))
    white_kings = np.count_nonzero(board_state == PieceType.WHITE_KING.value)
    black_kings = np.count_nonzero(board_state == PieceType.BLACK_KING.value)
    
    # Generate text description
    description = f"Current player: {game_instance.current_player.name}\n"
    description += f"Black pieces: {black_pieces} (including {black_kings} kings)\n"
    description += f"White pieces: {white_pieces} (including {white_kings} kings)\n"
    
    # Include the last move if available
    if game_instance.move_history:
        last_move = game_instance.move_history[-1]
        start_notation = utils.coord_to_notation(last_move['start'][0], last_move['start'][1])
        end_notation = utils.coord_to_notation(last_move['end'][0], last_move['end'][1])
        description += f"Last move: {last_move['player']} moved from {start_notation} to {end_notation}"
        if last_move['is_capture']:
            description += " (capture)"
        description += "\n"
    
    # Include available moves for current player
    description += f"Available moves for {game_instance.current_player.name}: {len(game_instance.available_moves)}\n"
    
    return description