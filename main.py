"""
Live Checkers Commentary System
===============================
A system that provides real-time commentary for checkers games using LLM technology.
"""
# python imports
import os
import logging
import traceback
from datetime import datetime

# local imports
from checkers_game import CheckersGame, CheckersUI
from checkers_evaluator import CheckersAnalyzer
from llm_integration import CommentaryGenerator
from config_util import get_config


# Get current date and time
now = datetime.now()

# Format the datetime object to a string
timestamp = now.strftime("%Y%m%d_%H%M%S")
log_dir = os.path.abspath('./')
log_file_name = f'game_log_{timestamp}.txt'
try:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_adr = os.path.join(log_dir, log_file_name)
    # Set up logging
    logging.basicConfig(
        filename=log_adr,
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    print(f"Logging to: {log_adr}")
except Exception as e:
    print(f"Error setting up logging: {str(e)}")
    # Fallback to console logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    log_adr = "console (file logging failed)"




# ===============================
# Checkers Game Logic
# ===============================

# ===============================
# Checkers AI/Analysis
# ===============================

# ===============================
# LLM Integration
# ===============================

# ===============================
# Text-to-Speech using system command
# ===============================

# ===============================
# UI Implementation
# ===============================

# ===============================
# Main Application
# ===============================

def main():
    """Main function to run the application."""
    try:
        logging.info("Starting application")
        
        # load config
        CONFIG = get_config()
        logging.info("Configuration loaded")
        logging.info(f"Config: {CONFIG}")
        # Initialize the game
        game = CheckersGame(CONFIG["board_size"])
        logging.info("Game initialized")
        
        # Initialize the analyzer
        analyzer = CheckersAnalyzer(game)
        logging.info("Analyzer initialized")
        
        # Initialize the commentator
        commentator = CommentaryGenerator(CONFIG["commentary_style"])
        logging.info("Commentator initialized")
        
        # Initialize the UI
        ui = CheckersUI(game, analyzer, commentator)
        logging.info("UI initialized")
        
        # Run the game
        ui.run()
        logging.info("Game run completed")
    except Exception as e:
        logging.error(f"Fatal error in main: {str(e)}")
        logging.error(traceback.format_exc())
        print(f"Fatal error: {str(e)}")
        print(f"Check {log_adr} for details")

if __name__ == "__main__":
    try:
        print("Starting Checkers Commentary System...")
        logging.info("===== APPLICATION STARTING =====")
        main()
    except Exception as e:
        logging.error(f"Unhandled exception: {str(e)}")
        logging.error(traceback.format_exc())
        print(f"Unhandled exception: {str(e)}")
        print(f"Check {log_adr} for details")
