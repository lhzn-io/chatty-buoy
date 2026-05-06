# Use the official NVIDIA PyTorch container for Thor (SBSA/R38)
FROM nvcr.io/nvidia/pytorch:26.02-py3

WORKDIR /app

# The following packages are omitted because PyTorch is already built-in
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    sse-starlette \
    aiohttp \
    redis \
    psutil \
    pydantic \
    openai \
    asyncpg \
    PyMuPDF \
    pgvector \
    sentence-transformers

ENV PYTHONPATH=/app
ENV L1_URI=http://front-end-service:8001/v1/chat/completions
ENV REDIS_HOST=redis

CMD ["uvicorn", "src.orchestrator.agent_orchestrator:app", "--host", "0.0.0.0", "--port", "8000"]
