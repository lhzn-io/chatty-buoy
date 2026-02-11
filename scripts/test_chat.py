#!/usr/bin/env python3
import argparse
import sys
import json
import urllib.request
import urllib.error

def stream_chat_completions(url, model, messages, max_tokens=1024, temperature=0.1):
    """
    Generator that yields (content_delta, reasoning_delta) from vLLM using urllib.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-dummy"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True 
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{url}/chat/completions", data=data, headers=headers)
    
    try:
        with urllib.request.urlopen(req) as response:
            for line in response:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0]["delta"]
                        
                        content = delta.get("content", "")
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning", "")
                        
                        yield content, reasoning
                    except json.JSONDecodeError:
                        continue
    except urllib.error.URLError as e:
        yield f"[Error: {e}]", None

def main():
    parser = argparse.ArgumentParser(description="Streaming Chat Client for Cortex")
    parser.add_argument("--url", default="http://localhost:8000/v1", help="Base URL")
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4", help="Model name")
    parser.add_argument("--system", default="You are a practical, grounded AI assistant. Be concise.", help="System prompt")
    parser.add_argument("--temp", type=float, default=0.1, help="Temperature")
    args = parser.parse_args()

    print(f"Connecting to {args.url}...")
    print(f"Model: {args.model}")
    print("Type 'quit', 'exit', or 'clear' to reset.\n")

    history = [
        {"role": "system", "content": args.system}
    ]

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            if user_input.lower() == "clear":
                history = [{"role": "system", "content": args.system}]
                print("History cleared.")
                continue
            
            history.append({"role": "user", "content": user_input})
            
            print("\nCortex: ", end="", flush=True)
            
            full_content = ""
            full_reasoning = ""
            is_reasoning = False
            has_printed_reasoning_header = False
            
            for content, reasoning in stream_chat_completions(args.url, args.model, history, temperature=args.temp):
                # Handle Reasoning (Inner Thoughts)
                if reasoning:
                    if not has_printed_reasoning_header:
                        print("\n[INNER THOUGHTS]\n", end="", flush=True)
                        has_printed_reasoning_header = True
                        is_reasoning = True
                    
                    print(reasoning, end="", flush=True)
                    full_reasoning += reasoning
                
                # Handle Content (Actual Answer)
                if content:
                    if is_reasoning:
                        print("\n\n[ANSWER]\n", end="", flush=True)
                        is_reasoning = False
                    
                    print(content, end="", flush=True)
                    full_content += content
            
            print("\n") # Newline at end
            
            # Save the regular content to history (ignoring reasoning for context to save tokens)
            history.append({"role": "assistant", "content": full_content})
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
