"""
Live Checkers Commentary System
===============================
A system that provides real-time commentary for checkers games using LLM technology.
"""
# library imports
from datetime import datetime
import config
import numpy as np
import requests
from checkers_game import CheckersGame, CheckersUI
from checkers_evaluator import CheckersAnalyzer
from config import CONFIG
from llm_integration import CommentaryGenerator
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
import utils


# Get current date and time
now = datetime.now()

# Format the datetime object to a string
timestamp = now.strftime("%Y%m%d_%H%M%S")
log_dir = './logs'
log_file_name = f'game_log_{timestamp}.log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_adr = os.path.join(log_dir, log_file_name)

# Set up logging
logging.basicConfig(
    filename=log_adr,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
print(f"Logging to: {log_adr}")




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
        CONFIG = config.get_config()
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
