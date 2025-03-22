
import logging
import subprocess
import sys


class CommandTTS:
    """A text-to-speech class that uses system commands."""
    
    def __init__(self):
        self.platform = sys.platform
        logging.info(f"Platform detected: {self.platform}")
        
    def speak(self, text):
        """Speak text using platform-specific commands."""
        # Create a safe version of the text for command line
        safe_text = text.replace('"', "'").strip()
        
        try:
            if self.platform == 'darwin':  # macOS
                # Use the built-in 'say' command on macOS
                cmd = ['say', safe_text]
                # Use subprocess.Popen and don't wait for completion
                subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                logging.info(f"Using macOS 'say' command: {safe_text[:30]}...")
                
            elif self.platform == 'win32':  # Windows
                # Use PowerShell's speak synthesis on Windows
                powershell_cmd = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{safe_text}");'
                cmd = ['powershell', '-Command', powershell_cmd]
                subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                logging.info(f"Using Windows PowerShell speech: {safe_text[:30]}...")
                
            elif self.platform.startswith('linux'):  # Linux
                # Try to use espeak on Linux if available
                cmd = ['espeak', f'"{safe_text}"']
                subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                logging.info(f"Using Linux espeak: {safe_text[:30]}...")
                
            else:
                logging.warning(f"TTS not supported on platform: {self.platform}")
                print(f"[TTS would say]: {text}")
                
        except Exception as e:
            logging.error(f"Error in TTS: {str(e)}")
            print(f"[TTS would say]: {text}")
    
    def stop(self):
        """Stop any currently playing speech."""
        try:
            if self.platform == 'darwin':  # macOS
                subprocess.call(['killall', 'say'])
                logging.info("Stopped macOS TTS")
            elif self.platform == "linux":
                subprocess.call(['killall', 'espeak'])
                logging.info("Stopped Linux TTS")
            # For other platforms, we don't have a simple way to stop speech
        except Exception as e:
            logging.error(f"Error stopping TTS: {str(e)}")
