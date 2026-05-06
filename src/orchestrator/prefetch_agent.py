
import asyncio
import json
import logging
import os
import time
from typing import List, Dict, Any

import redis.asyncio as redis
import asyncpg
from pgvector.asyncpg import register_vector
import aiohttp

from semantic_router.encoders import HuggingFaceEncoder

# --- Configuration ---
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
HISTORY_KEY = "agent_history"
CACHE_PREFIX = "semantic_cache:"

DB_CONFIG = {
    "user": "agent",
    "password": "agentbufferpassword",
    "database": "agent_memory",
    "host": "localhost",
    "port": 5432
}

L1_URI = "http://localhost:8001/v1/chat/completions"
L1_MODEL = "google/gemma-4-E4B-it"

# Threshold for similarity or top-k
TOP_K = 3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PrefetchAgent")

# Initialize Encoder (Snowflake-xs is 384 dim)
encoder = HuggingFaceEncoder(name="Snowflake/snowflake-arctic-embed-xs")

PREDICTION_PROMPT = """[PREDICTION TASK]
Based on the conversation history below, predict 3 specific technical nautical questions the Captain might ask next.
Focus on navigation rules, USCG requirements, or maritime safety.

HISTORY:
{history}

Return ONLY a JSON list of strings.
JSON:"""

class PrefetchAgent:
    """
    The 'Slow Thinker' background agent (Salesforce VoiceAgentRAG pattern).
    Predicts next questions and pre-warms the semantic cache.
    """
    def __init__(self):
        self.redis = None
        self.db_pool = None
        self.running = True

    async def connect(self):
        self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.db_pool = await asyncpg.create_pool(**DB_CONFIG)
        async with self.db_pool.acquire() as conn:
            await register_vector(conn)
        logger.info("Prefetch Agent Connected to Redis and Postgres.")

    async def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding using local HuggingFace model."""
        loop = asyncio.get_running_loop()
        vector = await loop.run_in_executor(None, encoder, [text])
        return vector[0]

    async def predict_questions(self, history: List[Dict]) -> List[str]:
        """Uses L1 to guess the next turn."""
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-5:]])
        prompt = PREDICTION_PROMPT.format(history=history_str)
        
        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "model": L1_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": 0.3
                }
                async with session.post(L1_URI, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data['choices'][0]['message']['content']
                        # Basic JSON cleanup
                        clean_content = content.replace("```json", "").replace("```", "").strip()
                        return json.loads(clean_content)
            except Exception as e:
                logger.error(f"Prediction Error: {e}")
        return []

    async def fetch_and_cache(self, query: str):
        """Perform RAG and push to Redis."""
        vector = await self._get_embedding(query)
        
        async with self.db_pool.acquire() as conn:
            # Semantic search using pgvector
            results = await conn.fetch(
                "SELECT content FROM documents ORDER BY embedding <=> $1 LIMIT $2",
                vector, TOP_K
            )
            
            if results:
                combined_context = "\n".join([r['content'] for r in results])
                # Store in Redis with TTL (e.g., 5 minutes)
                # Key is based on the query for semantic lookup
                # In a real system, we'd use an embedding-based cache lookup.
                # For this MVP, we store it under a general 'prefetch' key or turn-based key.
                await self.redis.set(f"{CACHE_PREFIX}latest", combined_context, ex=300)
                logger.info(f"Pre-fetched and Cached context for query: {query[:30]}...")

    async def run_loop(self):
        """Main background loop."""
        await self.connect()
        last_processed_idx = -1
        
        while self.running:
            try:
                # 1. Check history in Redis
                history_len = await self.redis.llen(HISTORY_KEY)
                if history_len > 0 and history_len - 1 > last_processed_idx:
                    # New turn detected
                    raw_history = await self.redis.lrange(HISTORY_KEY, 0, -1)
                    history = [json.loads(m) for m in raw_history]
                    last_processed_idx = history_len - 1
                    
                    # 2. Predict next questions
                    predicted = await self.predict_questions(history)
                    logger.info(f"Predicted Questions: {predicted}")
                    
                    # 3. Fetch and Cache
                    for q in predicted:
                        await self.fetch_and_cache(q)
                
                await asyncio.sleep(2) # Poll every 2s
            except Exception as e:
                logger.error(f"Loop Error: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    agent = PrefetchAgent()
    asyncio.run(agent.run_loop())
