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
        print(f"\n🚨 [PROACTIVE ALERT]: {text}\nUser > ", end="", flush=True)
        
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
                    print(chunk, end="", flush=True)
                
                client.stream_agent_response(text=text, callback_fn=print_chunk)
                print()
                
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    main()
