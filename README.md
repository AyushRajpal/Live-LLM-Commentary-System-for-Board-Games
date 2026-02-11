# Live LLM Commentary System for Board Games (Checkers → Chess)

An interactive board-game environment that generates **engine-grounded, real-time commentary** using Large Language Models (LLMs), with optional **text-to-speech (TTS)**.

This project began with **Checkers** and then transitioned to a stronger **Chess** implementation once the pipeline was validated, leveraging robust chess tooling (e.g., Stockfish) and richer domain knowledge in LLMs.

## Project Report
- 📄 **Paper / Report:** [Live-LLM-Commentary-System-for-Board-Games.pdf](docs/Live-LLM-Commentary-System-for-Board-Games.pdf)

## What this system does
At a high level, the system runs a loop each move:
1. **Extract board state** (structured summary / representation)
2. **Evaluate the position**
   - Checkers: heuristic + search (minimax with alpha-beta pruning)
   - Chess: engine-based evaluation (e.g., Stockfish) with top continuations
3. **Generate commentary** by prompting an LLM with:
   - current position + evaluation signals
   - (chess) best move + eval score + multi-line continuations, plus context about the previous move
4. **Render UI + output commentary** in real-time (plus optional TTS)

## Modules
### 1) Checkers (prototype in `main`)
- Playable 2-player checkers UI
- Game-state evaluation + LLM commentary + optional TTS
- This repo’s current `main` branch contains the checkers prototype:
  - `checkers_commentary.py`

### 2) Chess (experimental extension described in the report)
- A parallel chess implementation designed to take advantage of:
  - robust evaluation tooling (Stockfish)
  - LLM familiarity with chess concepts and terminology
- The report covers the chess prompt design and engine/LLM integration approach.

> Note: If your chess code lives on a separate branch, you can link it here (e.g., “See branch: `...`”).

## Tech Stack
- **Python**
- **UI:** `pygame`
- **LLM integration:** designed to be model/provider-agnostic (API-based or local runner)
- **TTS:** optional speech output (e.g., `pyttsx3` or OS-native commands)

## Requirements
- Python 3.8+ recommended
- Packages (current checkers prototype): `numpy`, `pygame`, `pyttsx3`, `requests`

## Quick Start (Checkers)
```bash
pip install numpy pygame pyttsx3 requests
python checkers_commentary.py
```

USC team project by:
Arad Firouzkouhi, Ayush Rajpal, Myesha Choudhury, Yash Bengali, Tyler Randall, Jonathan Reyhan, Yung-Chi Tsao
