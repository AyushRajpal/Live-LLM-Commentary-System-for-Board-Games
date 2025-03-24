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
import sys

# local imports
from checkers_game import CheckersGame, CheckersUI
from checkers_evaluator import CheckersAnalyzer
from llm_integration import CommentaryGenerator
from config_util import get_config

def setup_ollama_model():
    """ Load the OLLAMA model for generating commentary """
    CONFIG = get_config()
    if not CONFIG["llm_provider"].startswith("ollama:"):
        return
    model = CONFIG["llm_provider"].removeprefix("ollama:")
    try:
        import ollama
        ollama.generate(model=model, prompt="Test prompt",keep_alive="-1m")
    except Exception as e:
        logging.error(f"Error loading OLLAMA model: {str(e)}")
        logging.error(traceback.format_exc())
        print(f"Error loading OLLAMA model: {str(e)}")
        print("Check logs for details")
        


def setup_logging():
    """Set up logging with proper error handling and path resolution"""
    # Get current date and time
    now = datetime.now()
    
    # Format the datetime object to a string
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    # Use absolute path for logs directory - store in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs")
    log_file_name = f'game_log_{timestamp}.log'
    
    try:
        # Create logs directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_path = os.path.join(log_dir, log_file_name)
        
        # Check if we can write to the log file
        try:
            # Test if file can be opened for writing
            with open(log_path, 'w') as f:
                pass
            
            # Reset logging in case it was configured elsewhere
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
                
            # Set up logging with the verified path
            logging.basicConfig(
                filename=log_path,
                level=logging.DEBUG,
                format='%(asctime)s - %(levelname)s - %(message)s',
                filemode='w'
            )
            print(f"Logging to: {log_path}")
            return log_path
            
        except (IOError, PermissionError) as write_error:
            print(f"Cannot write to log file: {str(write_error)}")
            raise
            
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        print(f"Falling back to console logging")
        
        # Ensure no handlers remain from earlier attempts
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        # Configure for console logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )
        return "console (file logging failed)"


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
        log_adr = setup_logging()
        logging.info("===== APPLICATION STARTING =====")
        main()
    except Exception as e:
        logging.error(f"Unhandled exception: {str(e)}")
        logging.error(traceback.format_exc())
        print(f"Unhandled exception: {str(e)}")
        print(f"Check {log_adr} for details")
