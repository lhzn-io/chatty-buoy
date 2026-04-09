from openai import AsyncOpenAI
import os
from src.cortex.rag import search_docs

class CortexClient:
    def __init__(self, base_url="http://localhost:8000/v1"):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="EMPTY" # vLLM doesn't require auth locally
        )
        self.model = "allenai/Olmo-3-7B-Think"

    async def think(self, prompt: str) -> str:
        """
        Send a prompt to the Cortex and get a text response, augmented by RAG if applicable.
        """
        try:
            used_rag = False
            system_prompt = "You are Chatty-Buoy, a helpful AI crew member on a boat."
            
            # Fetch relevant documents using vector search
            docs = await search_docs(prompt, top_k=3)
            
            if docs:
                used_rag = True
                context_str = "\n\n".join([f"Source: {d['metadata']['source']} (Page {d['metadata']['page']})\n{d['content']}" for d in docs])
                system_prompt += "\n\nUse the following reference knowledge to answer the user's query accurately. Give a detailed and professional summary of the findings.\n\n" + context_str
                
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.4
            )
            
            result_text = response.choices[0].message.content
            print(f"[Cortex] RAG Sources Used: {used_rag} (Found: {len(docs)}) | Payload length: {len(result_text)}")
            return result_text
        except Exception as e:
            print(f"Cortex Error: {e}")
            return "My cortex is disconnected or failed to parse the archives."
