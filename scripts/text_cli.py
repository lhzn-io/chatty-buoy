import json
import sys
import requests
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

ORCHESTRATOR_URL = "http://localhost:8000/v1/chat/completions"

def main():
    print("📡 Connecting to Agent Orchestrator API (REST)...")
    
    session = PromptSession()
    
    while True:
        try:
            with patch_stdout():
                text = session.prompt("\nUser > ")
            
            if text.strip().lower() in ['exit', 'quit']:
                break
                
            if text.strip():
                print("\n🤖 Agent > ", end="", flush=True)
                
                payload = {
                    "model": "google/gemma-4-E4B-it",
                    "messages": [{"role": "user", "content": text}],
                    "stream": True
                }
                
                try:
                    response = requests.post(ORCHESTRATOR_URL, json=payload, stream=True)
                    response.raise_for_status()
                    
                    for line in response.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            if line.startswith("data: ") and line != "data: [DONE]":
                                try:
                                    data = json.loads(line[6:])
                                    chunk = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                    print(chunk, end="", flush=True)
                                except json.JSONDecodeError:
                                    pass
                    print()
                except requests.exceptions.RequestException as e:
                    print(f"\n❌ Error connecting to Orchestrator: {e}")
                
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    main()
