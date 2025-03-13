# Live Checkers Commentary System

A Python-based system that provides real-time commentary for checkers games using LLM technology.

## Overview

This project creates a checkers game with live commentary, combining:
- A complete checkers game with proper rules
- AI analysis of the game state
- LLM-generated commentary
- Text-to-speech to narrate the commentary

## Requirements

- Python 3.6+
- Required packages: numpy, pygame, pyttsx3, requests

## Quick Start

1. Install dependencies:
```bash
pip install numpy pygame pyttsx3 requests
```

2. Run the game:
```bash
python checkers_commentary.py
```

## How to Play

- **Game Rules**:
  - Black moves first
  - Regular pieces move diagonally forward only
  - Kings (with crown symbol) move diagonally in any direction
  - Capturing opponent pieces is mandatory when available

- **Controls**:
  - Click on a piece to select it
  - Valid moves will be highlighted in yellow
  - Click on a highlighted square to move there

## LLM Integration

By default, the system uses mock commentary. To use actual LLM commentary:

1. Get a Gemini API key from Google AI Studio
2. Update the CONFIG dictionary in checkers_commentary.py:
```python
CONFIG = {
    "llm_provider": "gemini",  # Change from "mock" to "gemini"
    "api_key": "YOUR_API_KEY_HERE",  # Add your Gemini API key
}
```

## Customization

Modify these values in the CONFIG dictionary:
- `commentary_style`: "educational", "casual", or "expert"
- `text_to_speech`: True or False
- `board_size`: Standard is 8, but can be changed

## Troubleshooting

- **Display Issues**: Adjust window size in CONFIG dictionary
- **Text-to-Speech Issues**: Try disabling it by setting `"text_to_speech": False`
