import asyncio
import os
import glob
import json
import hashlib
from typing import List, Dict, Any
import logging
import asyncpg
from pgvector.asyncpg import register_vector
from semantic_router.encoders import HuggingFaceEncoder

# CONFIG
DB_CONFIG = {
    "user": "agent",
    "password": "agentbufferpassword",
    "database": "agent_memory",
    "host": "localhost",
    "port": 5432
}
PDF_DIR = "./pdfs"
EMBED_DIM = 384 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG-Ingest")

# Initialize Encoder (Snowflake-xs is 384 dim)
encoder = HuggingFaceEncoder(name="Snowflake/snowflake-arctic-embed-xs")

async def get_embedding(text: str) -> List[float]:
    """Generate embedding using local HuggingFace model."""
    loop = asyncio.get_running_loop()
    # Snowflake-xs returns list of lists
    vector = await loop.run_in_executor(None, encoder, [text])
    return vector[0]

async def parse_pdf(filepath: str) -> List[Dict[str, Any]]:
    """Extract text chunks from PDF."""
    chunks = []
    try:
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                chunks.append({
                    "content": text,
                    "metadata": {"source": os.path.basename(filepath), "page": i+1}
                })
    except Exception as e:
        logger.error(f"Error parsing {filepath}: {e}")
    return chunks

def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

async def init_db(pool):
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id bigserial PRIMARY KEY,
                content text,
                metadata jsonb,
                embedding vector(384)
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_files (
                filepath text PRIMARY KEY,
                file_hash text NOT NULL
            )
        """)
        logger.info("Database initialized.")

async def ingest_docs():
    """Main ingestion routine."""
    try:
        pool = await asyncpg.create_pool(**DB_CONFIG)
        await init_db(pool)
        async with pool.acquire() as conn:
             await register_vector(conn)
    except Exception as e:
        logger.error(f"DB Connection failed: {e}")
        return

    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDFs.")

    for pdf_file in pdf_files:
        current_hash = compute_file_hash(pdf_file)
        
        async with pool.acquire() as conn:
            # Check if file has already been ingested with same hash
            row = await conn.fetchrow("SELECT file_hash FROM processed_files WHERE filepath = $1", pdf_file)
            if row and row['file_hash'] == current_hash:
                logger.info(f"Skipping {pdf_file} (already processed, hash unchanged).")
                continue
            
            logger.info(f"Processing {pdf_file}...")
            chunks = await parse_pdf(pdf_file)
            
            # Delete old chunks for this file if it was modified
            if row:
                await conn.execute("DELETE FROM documents WHERE metadata->>'source' = $1", os.path.basename(pdf_file))
            
            for chunk in chunks:
                vector = await get_embedding(chunk["content"])
                await conn.execute(
                    "INSERT INTO documents (content, metadata, embedding) VALUES ($1, $2, $3)",
                    chunk["content"], json.dumps(chunk["metadata"]), vector
                )
            
            # Update processed_files record
            await conn.execute("""
                INSERT INTO processed_files (filepath, file_hash) 
                VALUES ($1, $2) 
                ON CONFLICT (filepath) DO UPDATE SET file_hash = EXCLUDED.file_hash
            """, pdf_file, current_hash)
            
        logger.info(f"Ingested {len(chunks)} chunks from {pdf_file}.")

    await pool.close()

async def search_docs(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Search for relevant documents using vector similarity."""
    try:
        pool = await asyncpg.create_pool(**DB_CONFIG)
        async with pool.acquire() as conn:
            await register_vector(conn)
            query_vector = await get_embedding(query)
            
            # Using cosine distance <=> for similarity search
            rows = await conn.fetch(f"""
                SELECT content, metadata, 1 - (embedding <=> $1) AS similarity 
                FROM documents 
                ORDER BY embedding <=> $1 
                LIMIT $2
            """, query_vector, top_k)
            
            results = []
            for row in rows:
                metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                results.append({
                    "content": row['content'],
                    "metadata": metadata,
                    "similarity": float(row['similarity'])
                })
        await pool.close()
        return results
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []

if __name__ == "__main__":
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
    
    asyncio.run(ingest_docs())
