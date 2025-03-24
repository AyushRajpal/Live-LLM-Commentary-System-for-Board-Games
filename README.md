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
- For Linux users: `espeak` package is required for text-to-speech (install with `sudo apt-get install espeak`)

## Quick Start

1. Install dependencies:

```bash
pip install numpy pygame pyttsx3 requests
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

2. Run the game:

```bash
python main.py
```

## How to Play

- **Game Rules**:
  - Black moves first
  - Regular pieces move diagonally forward only
  - Kings (with crown symbol) move diagonally in any direction
  - Capturing opponent pieces is mandatory when available
  - A piece becomes a king when it reaches the opposite end of the board
  - The game ends when one player captures all of the opponent's pieces or blocks them from moving
  - If a piece can capture more than one opponent piece in a single turn, it can and must do so

- **Controls**:
  - Click on a piece to select it
  - Valid moves will be highlighted in yellow
  - Click on a highlighted square to move there

## LLM Integration

### Gemini API

By default, the system uses mock commentary. To use actual LLM commentary:

1. Get a Gemini API key from Google AI Studio
2. Update the CONFIG dictionary in config.json:

```json
{
    "llm_provider": "gemini",
    "api_key": "<Your API Key>",
}
```

Note that if the config file is not present, it will create a new one with default values, found in `config_util.py`.

### Ollama Local Model

Install Ollama through [this link](https://ollama.com/download).

Next, pull the desired model by running the following command in the terminal:

```bash
ollama pull <model>
```

For example, to pull the `gemma:4b` model, run:

```bash
ollama pull gemma:4b
```

To use the Ollama local model, set the `llm_provider` to `ollama:<model>` (e.g. `ollama:gemma:4b`) in the `config.json` file.

You may also need to run the Ollama server with the following command:

```bash
ollama serve
```

You can see a list of all models [here](https://ollama.com/search). It is recommended to use a small model in order to reduce the time it takes to generate commentary.

## Customization

Modify these values in the `config.json` file:

- `commentary_style`: "educational", "casual", or "expert"
- `text_to_speech`: True or False
- `board_size`: Standard is 8, but can be changed

## Troubleshooting

- **Display Issues**: Adjust window size in CONFIG dictionary
- **Text-to-Speech Issues**: Try disabling it by setting `"text_to_speech": False`
