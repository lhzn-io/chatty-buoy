import sys
import threading
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
import os
import sys

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.cortex.api_client import ChattyBuoyClient

def main():
    print("📡 Connecting to Agent Orchestrator API (REST)...")
    client = ChattyBuoyClient()
    
    def alert_callback(text):
        # Strip tags for alert printing
        clean_text = text.replace("<cite>", "").replace("</cite>", "")
        print(f"\n🚨 [PROACTIVE ALERT]: {clean_text}\nUser > ", end="", flush=True)
        
    threading.Thread(target=client.listen_for_alerts, args=(alert_callback,), daemon=True).start()
    
    session = PromptSession()
    
    while True:
        try:
            with patch_stdout():
                text = session.prompt("\nUser > ")
            
            if text.strip().lower() in ['exit', 'quit']:
                break
                
            if text.strip():
                print("\n🤖 Agent > ", end="", flush=True)
                
                def print_chunk(chunk):
                    # Strip tags but keep content in text mode
                    clean = chunk.replace("<cite>", "").replace("</cite>", "")
                    print(clean, end="", flush=True)
                
                client.stream_agent_response(text=text, callback_fn=print_chunk)
                print()
                
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    main()
