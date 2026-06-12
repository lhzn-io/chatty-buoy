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
------------------------------

The system features a containerized **audio-cli** service that handles the hardware audio interaction loop. This ensures unified management and logging via Docker.

**Start the Agent Interface**

Open a terminal and run the stack controller:

.. code-block:: bash

   ./scripts/stack.sh start

This will automatically launch the full cascade, including the audio loop.

**Monitoring the Audio Loop**

To view live audio processing logs and Quint's transcriptions:

.. code-block:: bash

   docker compose logs -f audio-cli

*   **Speak**: The system listens via the mapped Jabra Speak 710 hardware.
*   **Interaction**: Reports and alerts will stream to the dashboard and text-cli simultaneously.

4. Configure Watchstander Vision (Sentinel Mode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The system includes a containerized Watchstander component powered by **Cosmos-Reason2-2B**. It maintains a rolling video buffer and automatically interprets complex scenes (Sentinel Mode) using DeepStream bounding boxes when whitelisted objects (like people or vessels) are detected, pushing insights to the agent's memory via Redis.

**Dashboard**: Once `docker compose up -d` is running, the Watchstander Web Dashboard is available at `http://localhost:8080`.
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

Model Management and Offline Mode
---------------------------------

To ensure fast and reliable startups, especially in environments with limited or intermittent connectivity, the system is configured to run in **Offline Mode** by default and uses **Absolute Path Pinning** to ensure stability.

**Core Concepts:**

*   **HF_HUB_OFFLINE=1**: This environment variable prevents the system from reaching out to the Hugging Face API at startup. This eliminates network latency and prevents the service from hanging if the internet is slow or unavailable.
*   **Absolute Path Pinning**: Instead of referencing a model by its repo ID (e.g., ``google/gemma-4-E4B-it``), we point the vLLM engine directly to the specific snapshot directory on the local disk. This acts as a strong version pin, preventing accidental regressions if a model provider updates the weights on the hub.

**Current Configuration Example (from ``docker-compose.yaml``):**

.. code-block:: yaml

   front-end-service:
     environment:
       - HF_HUB_OFFLINE=1
       - HF_HOME=/root/.cache/huggingface
     entrypoint: [
       "vllm", "serve", "/root/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/292a7e278a400932df35f9fd4b1501edd04133a5",
       ...
     ]

   cosmos-vision:
     environment:
       - HF_HUB_OFFLINE=1
       - HF_HOME=/data/models/huggingface
     volumes:
       - ~/.cache/huggingface:/data/models/huggingface
     # Note: vLLM 0.14.0 requires Repo ID; cache is resolved via HF_HOME
     entrypoint: [ "vllm", "serve", "nvidia/Cosmos-Reason2-2B", ... ]

Administrative Procedures
^^^^^^^^^^^^^^^^^^^^^^^^^

These tasks should only be performed by an intentional operator when a network connection is available.

**1. Check for Model Updates**
To check if a newer version of a model is available:

1.  Temporarily edit ``docker-compose.yaml`` to set ``HF_HUB_OFFLINE=0``.
2.  Change the model path back to the Repo ID (e.g., ``google/gemma-4-E4B-it``).
3.  Restart the service (``docker compose up -d <service>``).
4.  If a new version is downloaded, vLLM will automatically create a new snapshot folder.

**2. Force a Model Snapshot Update**
If you want to move the entire stack to a newer version:

1.  Follow the steps above to download the new snapshot.
2.  Find the new snapshot hash by inspecting the local cache:
    ``ls ~/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/``
3.  Update the ``entrypoint`` in ``docker-compose.yaml`` with the new absolute path.
4.  Set ``HF_HUB_OFFLINE=1`` again.
5.  Restart the stack.

**3. Troubleshooting "Unhealthy" Models**
The stack controller (``scripts/stack.sh``) includes self-healing logic. If a model service becomes unhealthy (due to memory issues or cache mismatch), run:

.. code-block:: bash

   ./scripts/stack.sh start

The script will automatically detect the ``unhealthy`` status and force a clean restart of that specific service.

