User Guide
==========

Thor Semantic Audio Agent
-------------------------

This system implements a production-grade, low-latency Voice AI Agent for the Jetson AGX Thor, featuring a "Semantic Audio" architecture (FP8 TTS + GStreamer Audio Pipeline).

System Components
^^^^^^^^^^^^^^^^^

*   **ASR**: ``asr-service`` (Riva / Parakeet-TDT-1.1B). *High-Speed Streaming ASR.*
*   **TTS**: ``tts-service`` (vLLM / CosyVoice2-0.5B / FP8). *Thor-Accelerated Synthesis.*
*   **Brain**: ``brain-service`` (Nemotron-3-Nano-30B-A3B-NVFP4). *Reasoning Engine.*
*   **Memory**: ``postgres-vector`` (PostgreSQL + pgvector). *Multimodal RAG.*
*   **Agent**: ``src/agent_reflex.py``. *GStreamer Orchestrator.*

Getting Started
---------------

1. Start the Stack
^^^^^^^^^^^^^^^^^^

Ensure your Environment is set up and Cached Models are mapped.

.. code-block:: bash

   docker compose up -d

Verify services are running:

.. code-block:: bash

   docker compose ps

2. Ingest Knowledge (RAG)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
   Ensure containers are running ``docker compose up -d`` **before** running ingestion, as it connects to the local Postgres port 5432.

Place your PDF documents in the ``./pdfs/`` directory.

Run the ingestion script (from the host):

.. code-block:: bash

   micromamba run -n chatty-buoy python3 src/rag_ingest.py

This will index the documents into the ``postgres-vector`` database.

3. Run the Agent (Hybrid Mode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Architecture Note:** 
To maximize performance on Jetson Thor (ARM64) and avoid emulation overhead, we use a hybrid approach:
*   **Infrastructure**: Docker containers for ASR (Riva), Brain (Triton), and Memory (Postgres).
*   **TTS**: Local Native Execution for CosyVoice2 (Python/PyTorch) to leverage the GPU directly without x86 container definitions.

**Step 3.1: Start TTS Server (Local)**

Open a new terminal, activate the environment, and run the launcher:

.. code-block:: bash

   micromamba activate chatty-buoy
   bash scripts/start_tts.sh

*This script will ensure dependencies (`cosyvoice`, `fastapi`, `uvicorn`) are installed and launch the API server on port 50000.*

**Step 3.2: Start Agent Interface**

In your main terminal:

.. code-block:: bash

   micromamba run -n chatty-buoy python3 src/orchestrator/main.py

*   **Speak**: The system listens via ALSA (Default Device / Jabra).
*   **Reflex**: Simple queries or conversational fillers are handled immediately.
4. Configure Watchstander Vision (Sentinel Mode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The system includes a containerized Watchstander component powered by **Cosmos-Reason2-2B**. It maintains a rolling video buffer and automatically interprets complex scenes (Sentinel Mode) using DeepStream bounding boxes when whitelisted objects (like people or vessels) are detected, pushing insights to the agent's memory via Redis.

**Dashboard**: Once `docker compose up -d` is running, the Sentinel Web Dashboard is available at `http://localhost:8080`.
Through the dashboard, you can monitor the live event feed, review Cosmos reasoning clips, and dynamically tune the VLM prompt configurations without restarting the service.

You can configure the video source using the ``RTSP_URL`` variable in ``docker-compose.yaml`` under the ``vision-service`` (or ``cosmos-vision``) configurations:

*   **Physical IP Webcam**:
    Pass the direct RTSP stream URL to ``RTSP_URL``:

    .. code-block:: yaml

       environment:
         - RTSP_URL=rtsp://192.168.3.243:8080/video

*   **Direct USB HD Camera (plugged into Thor)**:
    Mount the hardware device and set ``RTSP_URL`` to the local index (``0``).

    .. code-block:: yaml

       # In docker-compose.yaml under vision-service:
       devices:
         - "/dev/video0:/dev/video0"
       environment:
         - RTSP_URL=0

*   **Live YouTube Video Stream (Simulator)**:
    Uncomment the ``mediamtx`` and ``rtsp-simulator`` services in your docker-compose file. Set ``YOUTUBE_URL``, and point the vision service to the local media server:

    .. code-block:: yaml

       environment:
         - RTSP_URL=rtsp://mediamtx:8554/live
