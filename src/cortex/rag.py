import asyncio
import os
import glob
import json
import hashlib
from typing import List, Dict, Any, Union
import logging
import asyncpg
import fitz # PyMuPDF
from PIL import Image
import io
from pgvector.asyncpg import register_vector
from sentence_transformers import SentenceTransformer, CrossEncoder

# CONFIG
DB_CONFIG = {
    "user": "agent",
    "password": "agentbufferpassword",
    "database": "agent_memory",
    "host": os.environ.get("PG_HOST", "localhost"),
    "port": 5432
}
PDF_DIR = "./pdfs"
EMBED_DIM = 512  # clip-ViT-B-32 is 512 dim
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG-Ingest")

# Initialize Multimodal Encoder (OpenAI CLIP ViT-B/32 encodes both text and images to 512-dim natively)
model = None
# Initialize CrossEncoder for text Reranking to fix CLIP's semantic drift
reranker = None

# Enable/Disable RAG feature with an environment toggle (default to False to prevent unwanted startup downloads)
ENABLE_RAG = os.environ.get("ENABLE_RAG", "false").lower() in ("true", "1", "yes")

def _lazy_init_models():
    global model, reranker
    if not ENABLE_RAG:
        return
    if model is None:
        logger.info("Lazy-loading SentenceTransformer model...")
        model = SentenceTransformer("clip-ViT-B-32")
    if reranker is None:
        logger.info("Lazy-loading CrossEncoder model...")
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

async def get_embedding(content: Union[str, Image.Image]) -> List[float]:
    _lazy_init_models()
    if model is None:
        return [0.0] * EMBED_DIM
    loop = asyncio.get_running_loop()
    vector = await loop.run_in_executor(None, lambda: model.encode(content))
    return vector.tolist()

async def parse_pdf_multimodal(filepath: str) -> List[Dict[str, Any]]:
    chunks = []
    try:
        doc = fitz.open(filepath)
        for i, page in enumerate(doc):
            text = page.get_text()
            if text and text.strip():
                text_clean = text.strip()
                # Semantic Text Chunking (sliding window) to prevent diluted vectors
                start = 0
                while start < len(text_clean):
                    end = start + CHUNK_SIZE
                    chunk_text = text_clean[start:end]
                    if len(chunk_text.strip()) > 50:
                        chunks.append({
                            "type": "text",
                            "content": chunk_text.strip(),
                            "metadata": {"source": os.path.basename(filepath), "page": i+1, "type": "text"}
                        })
                    start += (CHUNK_SIZE - CHUNK_OVERLAP)
            
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                try:
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    if image.width < 100 or image.height < 100:
                        continue
                    chunks.append({
                        "type": "image",
                        "content": image,
                        "metadata": {"source": os.path.basename(filepath), "page": i+1, "type": "image", "img_index": img_index}
                    })
                except Exception as e:
                    logger.error(f"Error reading image on page {i+1}: {e}")
    except Exception as e:
        logger.error(f"Error parsing {filepath}: {e}")
    return chunks

def compute_file_hash(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

async def init_db(pool):
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        try:
            await conn.execute("ALTER TABLE documents ALTER COLUMN embedding TYPE vector(512)")
            logger.info("Altered documents embedding column to 512 dimensions.")
        except Exception:
            logger.info("Dropping table documents to recreate with 512 dimensions.")
            await conn.execute("DROP TABLE IF EXISTS documents")
            await conn.execute("""
                CREATE TABLE documents (
                    id bigserial PRIMARY KEY,
                    content text,
                    metadata jsonb,
                    embedding vector(512)
                )
            """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_files (
                filepath text PRIMARY KEY,
                file_hash text NOT NULL
            )
        """)

async def ingest_docs():
    if not ENABLE_RAG:
        logger.info("RAG is disabled. Skipping document ingestion.")
        return

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
            row = await conn.fetchrow("SELECT file_hash FROM processed_files WHERE filepath = $1", pdf_file)
            if row and row['file_hash'] == current_hash:
                logger.info(f"Skipping {pdf_file} (already processed, hash unchanged).")
                continue
            
            logger.info(f"Processing {pdf_file}...")
            chunks = await parse_pdf_multimodal(pdf_file)
            
            if row:
                await conn.execute("DELETE FROM documents WHERE metadata->>'source' = $1", os.path.basename(pdf_file))
            
            for chunk in chunks:
                vector = await get_embedding(chunk["content"])
                db_content = chunk["content"] if chunk["type"] == "text" else f"[IMAGE Extracted from {os.path.basename(pdf_file)} Page {chunk['metadata']['page']}]"
                db_content = db_content.replace('\x00', '') # Sanitize illegal Postgres null bytes
                await conn.execute(
                    "INSERT INTO documents (content, metadata, embedding) VALUES ($1, $2, $3)",
                    db_content, json.dumps(chunk["metadata"]), vector
                )
            
            await conn.execute("""
                INSERT INTO processed_files (filepath, file_hash) 
                VALUES ($1, $2) 
                ON CONFLICT (filepath) DO UPDATE SET file_hash = EXCLUDED.file_hash
            """, pdf_file, current_hash)
            
        logger.info(f"Ingested {len(chunks)} multimodal chunks from {pdf_file}.")
    await pool.close()

async def search_docs(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    if not ENABLE_RAG:
        return []
        
    try:
        pool = await asyncpg.create_pool(**DB_CONFIG)
        async with pool.acquire() as conn:
            await register_vector(conn)
            query_vector = await get_embedding(query)
            
            # 1. Broad retrieval (First stage)
            candidate_k = max(top_k * 5, 15)
            rows = await conn.fetch(f"""
                SELECT id, content, metadata, 1 - (embedding <=> $1) AS similarity 
                FROM documents 
                ORDER BY embedding <=> $1 
                LIMIT $2
            """, query_vector, candidate_k)
            
            candidates = []
            for row in rows:
                metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                candidates.append({
                    "id": row['id'],
                    "content": row['content'],
                    "metadata": metadata,
                    "similarity": float(row['similarity'])
                })
                
        await pool.close()
        
        if not candidates:
            return []
            
        # 2. Local Cross-Encoder Reranking (Second stage)
        loop = asyncio.get_running_loop()
        pairs = [[query, doc['content']] for doc in candidates]
        
        # Execute blocking reranker model call in a thread
        scores = await loop.run_in_executor(None, lambda: reranker.predict(pairs))
        
        # Update scores and sort
        for i, score in enumerate(scores):
            candidates[i]['rerank_score'] = float(score)
            
        # Sort descending by the highly precise rerank score
        candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        return candidates[:top_k]
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []

if __name__ == "__main__":
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
    asyncio.run(ingest_docs())
