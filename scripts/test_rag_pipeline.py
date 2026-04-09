import asyncio
import sys
from src.cortex.rag import search_docs
from src.cortex.client import CortexClient

async def test_rag(query: str):
    print(f"\n🌊 [PHASE 1] HITTING PGVECTOR DIRECTLY...")
    print(f"Query: '{query}'\n")
    
    docs = await search_docs(query, top_k=3)
    if not docs:
        print("❌ No matching chunks found in PostgreSQL.")
        return
        
    for i, res in enumerate(docs, 1):
        source = res['metadata'].get('source', 'Unknown')
        page = res['metadata'].get('page', '?')
        content_preview = res['content'].replace('\n', ' ')[:200]
        print(f"  [{i}] Source: {source} (Page {page})")
        print(f"      Similarity: {res['similarity']:.3f} | Content: {content_preview}...")
        
    print(f"\n🧠 [PHASE 2] HITTING CORTEX-ENGINE (Olmo-3) WITH RAG CONTEXT...")
    cortex = CortexClient()
    
    print(f"Sending formatted payload to LLM at {cortex.client.base_url}...\n")
    try:
        response = await cortex.think(query)
        print("====== CORTEX RESPONSE ======\n")
        print(response)
        print("\n=============================\n")
    except Exception as e:
        print(f"❌ Failed to reach Cortex-Engine: {e}")

if __name__ == "__main__":
    test_query = "What is the best way to handle an alteration of course to avoid a close-quarters situation if we have sufficient sea room?"
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])
    
    asyncio.run(test_rag(test_query))
