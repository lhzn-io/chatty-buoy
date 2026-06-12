import json
import logging
import requests

logger = logging.getLogger("ApiClient")

class ChattyBuoyClient:
    def __init__(self, orchestrator_host="localhost", orchestrator_port=8000):
        self.orchestrator_url = f"http://{orchestrator_host}:{orchestrator_port}/v1/chat/completions"
        self.events_url = f"http://{orchestrator_host}:{orchestrator_port}/v1/events/stream"

    def listen_for_alerts(self, callback_fn, interrupt_event=None):
        """
        Listen for autonomous proactive updates from the orchestrator over SSE.
        """
        try:
            logger.info("Listening to Orchestrator SSE stream for proactive alerts.")
            with requests.get(self.events_url, stream=True) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if interrupt_event and interrupt_event.is_set():
                        continue
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith("data: "):
                            data = line[6:]
                            try:
                                payload = json.loads(data)
                                text = payload.get("text", "")
                            except Exception:
                                text = data
                            if text:
                                callback_fn(text)
        except Exception as e:
            logger.error(f"SSE listener failed: {e}")

    def stream_agent_response(self, text=None, b64_audio=None, callback_fn=None, interrupt_event=None, first_token_callback=None, transcript_callback=None):
        """
        Stream a response from the Orchestrator. Provide either text or b64_audio.
        """
        # Construct the payload
        messages = [{"role": "user"}]
        if b64_audio:
            messages[0]["content"] = [
                {"type": "input_audio", "input_audio": {"data": b64_audio, "format": "wav"}}
            ]
        elif text:
            messages[0]["content"] = text
        else:
            raise ValueError("Must provide either text or b64_audio")

        payload = {
            "model": "google/gemma-4-E4B-it",
            "messages": messages,
            "stream": True,
            "enable_intent_filter": True if b64_audio else False
        }

        try:
            # Use a longer timeout for the streaming response itself, 
            # but iter_lines will still block if the server is silent.
            with requests.post(self.orchestrator_url, json=payload, stream=True, timeout=(5, 120)) as resp:
                resp.raise_for_status()
                first_token_received = False
                for line in resp.iter_lines(delimiter=b'\n'):
                    if interrupt_event and interrupt_event.is_set():
                        break
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                data = json.loads(line[6:])
                                if data.get("type") == "transcript":
                                    if transcript_callback:
                                        transcript_callback(data.get("text", ""))
                                    continue
                                chunk = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if chunk:
                                    if not first_token_received:
                                        first_token_received = True
                                        if first_token_callback:
                                            first_token_callback()
                                    if callback_fn:
                                        callback_fn(chunk)
                            except json.JSONDecodeError:
                                pass
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Orchestrator: {e}")
