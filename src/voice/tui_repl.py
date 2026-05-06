import requests
import subprocess
import sys
import os
import threading
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style

# Ensure we can import chatty_buoy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from chatty_buoy.audio.jabra_config import find_jabra_device

# TTS API Configuration
TTS_URL = "http://localhost:8003/generate"
OUTPUT_FILE = "tui_live_output.raw"
VOICE_ID = "neutral_female_en"
MODEL_ID = "chatterbox"

style = Style.from_dict({
    'prompt': 'ansicyan bold',
    'status': 'ansiyellow',
    'error': 'ansired bold',
    'success': 'ansigreen bold'
})

def play_audio(file_path):
    try:
        # Chatterbox returns raw PCM audio. Using 'play' (SoX) to play the raw file.
        # Assuming 24kHz, 16-bit, 1-channel based on standard raw outputs.
        # We can also just pipe `aplay -f S16_LE -r 24000 -c 1`
        if sys.platform == 'darwin':
            subprocess.run(["play", "-t", "raw", "-r", "24000", "-e", "signed", "-b", "16", "-c", "1", file_path], check=True)
        else:
            jabra = find_jabra_device()
            cmd = ["aplay", "-q", "-f", "S16_LE", "-r", "24000", "-c", "1"]
            if jabra:
                # Use plughw which automatically resamples 24kHz to hardware rate
                cmd.extend(["-D", f"plughw:{jabra[0]},0"])
            cmd.append(file_path)
            subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Failed to play audio: {e}")

def generate_and_play(text):
    print(HTML(f"<status>Generating audio for:</status> {text}"))
    
    payload = {
        "text": text,
        "voice": VOICE_ID
    }

    try:
        resp = requests.post(TTS_URL, json=payload, stream=True)
        if resp.status_code == 200:
            print(HTML(f"<success>Streaming audio...</success>"))
            
            # Start the playback process waiting for stdin
            if sys.platform == 'darwin':
                cmd = ["play", "-t", "raw", "-r", "24000", "-e", "signed", "-b", "16", "-c", "1", "-"]
            else:
                jabra = find_jabra_device()
                # Use --buffer-size and --period-size to prevent underruns
                cmd = ["aplay", "-q", "-f", "S16_LE", "-r", "24000", "-c", "1", "--buffer-size=8192", "--period-size=2048"]
                if jabra:
                    cmd.extend(["-D", f"plughw:{jabra[0]},0"])
                cmd.append("-")
                
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            
            try:
                # Stream chunks directly into the player's stdin buffer
                for chunk in resp.iter_content(chunk_size=2048):
                    if chunk:
                        proc.stdin.write(chunk)
                        proc.stdin.flush()
            finally:
                proc.stdin.close()
                proc.wait()
                
        else:
            print(HTML(f"<error>Error {resp.status_code}: {resp.text}</error>"))
    except requests.exceptions.ConnectionError:
            print(HTML(f"<error>Connection failed. Is the TTS service running at {TTS_URL}?</error>"))
    except Exception as e:
        print(HTML(f"<error>Exception: {e}</error>"))

def main():
    print(HTML("<b>=== TTS TUI REPL ===</b>"))
    print(HTML(f"Connected to: <status>{TTS_URL}</status> | Voice: <status>{VOICE_ID}</status>"))
    jabra = find_jabra_device()
    if jabra:
        print(HTML(f"Audio Output: <status>Jabra Speaker (Card {jabra[0]})</status>"))
    else:
        print(HTML(f"Audio Output: <status>System Default</status>"))
        
    print("Type your message and press Enter to generate speech.")
    print("Type 'q', 'quit', or 'exit' to stop, or use Ctrl+C / Ctrl+D.\n")

    session = PromptSession()

    while True:
        try:
            with patch_stdout():
                text = session.prompt(HTML("<prompt>TTS> </prompt> "), style=style)
                
                text = text.strip()
                if not text:
                    continue
                    
                if text.lower() in ['q', 'quit', 'exit']:
                    break

                # Run generating and playing in a background thread to keep input responsive
                threading.Thread(target=generate_and_play, args=(text,), daemon=True).start()

        except KeyboardInterrupt:
            # Handle Ctrl+C
            continue
        except EOFError:
            # Handle Ctrl+D
            break

    print("Goodbye!")

if __name__ == "__main__":
    main()