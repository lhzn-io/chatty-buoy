
import asyncio
import json
import redis.asyncio as redis
import time

async def test_flow():
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    print("--- 🧪 Testing Proactive Pre-fetching Flow ---")
    
    # 1. Clear state
    await r.delete("agent_history")
    await r.delete("semantic_cache:latest")
    
    # 2. Simulate User Turn 1
    user_msg = {"role": "user", "content": "Quint, I'm planning a voyage on a 40-foot recreational vessel."}
    print(f"User: {user_msg['content']}")
    await r.rpush("agent_history", json.dumps(user_msg))
    
    print("Waiting for Prefetch Agent to predict and cache (5s)...")
    await asyncio.sleep(8)
    
    # 3. Check Cache
    cached = await r.get("semantic_cache:latest")
    if cached:
        print("✅ SUCCESS: Found pre-fetched context in cache!")
        print(f"Context Snippet: {cached[:200]}...")
    else:
        print("❌ FAILURE: No context found in cache.")
        # Check logs
        print("\nChecking prefetch_agent.log:")
        with open("prefetch_agent.log", "r") as f:
            print(f.read())

if __name__ == "__main__":
    asyncio.run(test_flow())
