from openai import OpenAI
import os

class CortexClient:
    def __init__(self, base_url="http://localhost:8000/v1"):
        self.client = OpenAI(
            base_url=base_url,
            api_key="EMPTY" # vLLM doesn't require auth locally
        )
        self.model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"

    def think(self, prompt: str) -> str:
        """
        Send a prompt to the Cortex and get a text response.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are Chatty-Buoy, a helpful AI crew member on a boat."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=256,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Cortex Error: {e}")
            return "My cortex is disconnected."
