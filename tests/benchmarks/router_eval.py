import json
import time
import asyncio
import requests
from typing import List, Dict

# High-Fidelity Synthetic Dataset (100 Scenarios)
# Optimized for Jetson Thor / VoiceAgentRAG Benchmark
DATASET = [
    # --- IGNORE (Ambient Passenger Chatter) ---
    {"input": "Pass me a cold one from the cooler, will you?", "category": "IGNORE"},
    {"input": "Did you see that splash over by the reef?", "category": "IGNORE"},
    {"input": "I think the tide is coming in, look at the shoreline.", "category": "IGNORE"},
    {"input": "Hand me the binoculars, Sarah.", "category": "IGNORE"},
    {"input": "Is it time for lunch yet? I'm starving.", "category": "IGNORE"},
    {"input": "The water is so clear today, you can see all the way to the bottom.", "category": "IGNORE"},
    {"input": "Don't forget to put on more sunscreen.", "category": "IGNORE"},
    {"input": "That's a massive yacht coming out of the harbor.", "category": "IGNORE"},
    {"input": "I should have brought my fishing pole.", "category": "IGNORE"},
    {"input": "Check out the clouds on the horizon, looks like a front.", "category": "IGNORE"},
    {"input": "Where did I put my sunglasses?", "category": "IGNORE"},
    {"input": "This sandwich is great, thanks for making it.", "category": "IGNORE"},
    {"input": "The engine sounds a bit different today, don't you think?", "category": "IGNORE"},
    {"input": "Hey, look at those dolphins jumping!", "category": "IGNORE"},
    {"input": "I'm going to take a nap on the deck.", "category": "IGNORE"},
    {"input": "We should head back before it gets dark.", "category": "IGNORE"},
    {"input": "Do we have enough bait for another hour?", "category": "IGNORE"},
    {"input": "Look at the wake we're leaving behind.", "category": "IGNORE"},
    {"input": "I love the smell of the salt air.", "category": "IGNORE"},
    {"input": "Did anyone bring the radio?", "category": "IGNORE"},
    {"input": "Pass the salt, please.", "category": "IGNORE"},
    {"input": "Talking to someone else right now.", "category": "IGNORE"},
    {"input": "Umm, wait, let me think...", "category": "IGNORE"},
    {"input": "Just some background noise.", "category": "IGNORE"},
    {"input": "Hey, pass me that rope.", "category": "IGNORE"},

    # --- FAST_CHAT (Quint Persona / General Interaction) ---
    {"input": "Quint, how's your day going so far?", "category": "FAST_CHAT"},
    {"input": "Tell me a joke about a pirate and a parrot.", "category": "FAST_CHAT"},
    {"input": "What's the best thing about being a semantic buoy?", "category": "FAST_CHAT"},
    {"input": "Good morning Quint, are you ready for the mission?", "category": "FAST_CHAT"},
    {"input": "Who is your favorite explorer?", "category": "FAST_CHAT"},
    {"input": "Quint, tell me a sea shanty.", "category": "FAST_CHAT"},
    {"input": "What do you know about the history of this harbor?", "category": "FAST_CHAT"},
    {"input": "Are you feeling functional today, Quint?", "category": "FAST_CHAT"},
    {"input": "Quint, what is the meaning of life at sea?", "category": "FAST_CHAT"},
    {"input": "Hello there, Assistant. Give me a greeting.", "category": "FAST_CHAT"},
    {"input": "Quint, do you ever get lonely out here?", "category": "FAST_CHAT"},
    {"input": "Tell me a story about a storm you've survived.", "category": "FAST_CHAT"},
    {"input": "What's your favorite nautical myth?", "category": "FAST_CHAT"},
    {"input": "Quint, say something inspiring for the crew.", "category": "FAST_CHAT"},
    {"input": "How many knots can you handle in a gale?", "category": "FAST_CHAT"},
    {"input": "Quint, what's your primary objective?", "category": "FAST_CHAT"},
    {"input": "Tell me a fun fact about Blackwell GPUs.", "category": "FAST_CHAT"},
    {"input": "Quint, are you recording our telemetry?", "category": "FAST_CHAT"},
    {"input": "What's the weather forecast for tomorrow?", "category": "FAST_CHAT"},
    {"input": "Quint, what do you think of Sarah's sailing skills?", "category": "FAST_CHAT"},
    {"input": "Tell me a joke about a buoy and a lighthouse.", "category": "FAST_CHAT"},
    {"input": "Quint, give me a quick welcome to the platform.", "category": "FAST_CHAT"},
    {"input": "Are you connected to the satellite link?", "category": "FAST_CHAT"},
    {"input": "Quint, define 'semantic audio agent'.", "category": "FAST_CHAT"},
    {"input": "How's the water temperature today?", "category": "FAST_CHAT"},

    # --- TOOL_CALL (Imperative Hardware/Nav Commands) ---
    {"input": "Quint, give me a status report on the Jetson's VRAM.", "category": "TOOL_CALL"},
    {"input": "Set a waypoint at 34 degrees North, 118 degrees West.", "category": "TOOL_CALL"},
    {"input": "What is the current CPU temperature?", "category": "TOOL_CALL"},
    {"input": "Check the battery levels of the secondary sensors.", "category": "TOOL_CALL"},
    {"input": "Identify that vessel on the starboard side.", "category": "TOOL_CALL"},
    {"input": "Quint, show me the power draw from the Blackwell module.", "category": "TOOL_CALL"},
    {"input": "Mark a waypoint at these coordinates: 33.5N, 117.9W.", "category": "TOOL_CALL"},
    {"input": "Run a system diagnostic on the vision pipeline.", "category": "TOOL_CALL"},
    {"input": "Quint, list all active waypoints.", "category": "TOOL_CALL"},
    {"input": "Check the network latency to the mainland.", "category": "TOOL_CALL"},
    {"input": "Quint, what is our current orientation and pitch?", "category": "TOOL_CALL"},
    {"input": "Set a waypoint for the 'Old Wreck' at 34.1N, 118.5W.", "category": "TOOL_CALL"},
    {"input": "Give me a summary of the latest telemetry logs.", "category": "TOOL_CALL"},
    {"input": "Quint, check if the solar arrays are charging.", "category": "TOOL_CALL"},
    {"input": "Calibrate the IMU sensors now.", "category": "TOOL_CALL"},
    {"input": "Quint, what is the status of the AIS receiver?", "category": "TOOL_CALL"},
    {"input": "Mark our current location as a waypoint named 'Home'.", "category": "TOOL_CALL"},
    {"input": "Check for any hardware alerts in the last hour.", "category": "TOOL_CALL"},
    {"input": "Quint, what's the fan speed on the Jetson?", "category": "TOOL_CALL"},
    {"input": "Set a safety waypoint 5 miles to the South.", "category": "TOOL_CALL"},
    {"input": "Give me the coordinates of the last detected object.", "category": "TOOL_CALL"},
    {"input": "Quint, verify the storage space on the SSD.", "category": "TOOL_CALL"},
    {"input": "Check the health of the acoustic modem.", "category": "TOOL_CALL"},
    {"input": "Quint, start a new waypoint track.", "category": "TOOL_CALL"},
    {"input": "Show me the real-time GPU load.", "category": "TOOL_CALL"},

    # --- DEEP_RAG (Technical/Legal Nautical Knowledge) ---
    {"input": "What are the specific requirements for life jackets on a 40-foot recreational vessel?", "category": "DEEP_RAG"},
    {"input": "Explain the difference between a cardinal mark and a lateral mark.", "category": "DEEP_RAG"},
    {"input": "What is the rule for overtaking a vessel in a narrow channel according to the USCG handbook?", "category": "DEEP_RAG"},
    {"input": "How do I calculate my position using a nautical chart and a hand-bearing compass?", "category": "DEEP_RAG"},
    {"input": "What does a flashing yellow light on a buoy indicate in the US Aids to Navigation system?", "category": "DEEP_RAG"},
    {"input": "Explain the Right of Way when a sailing vessel is overtaking a power-driven vessel.", "category": "DEEP_RAG"},
    {"input": "What are the lighting requirements for a vessel at anchor in a designated anchorage?", "category": "DEEP_RAG"},
    {"input": "Summarize the COLREGs rules for preventing collisions at sea.", "category": "DEEP_RAG"},
    {"input": "What is the meaning of a 'restricted in her ability to maneuver' signal?", "category": "DEEP_RAG"},
    {"input": "Explain the difference between true north and magnetic north in navigation.", "category": "DEEP_RAG"},
    {"input": "What are the legal requirements for reporting a boating accident to the USCG?", "category": "DEEP_RAG"},
    {"input": "Describe the proper procedure for using a VHF radio to make a distress call.", "category": "DEEP_RAG"},
    {"input": "What is the purpose of a DSC alert on a maritime radio?", "category": "DEEP_RAG"},
    {"input": "How do I interpret a 'danger' buoy in the Western Rivers system?", "category": "DEEP_RAG"},
    {"input": "What are the speed restrictions in a 'No Wake' zone according to federal law?", "category": "DEEP_RAG"},
    {"input": "Explain the use of 'rules of the road' in international waters.", "category": "DEEP_RAG"},
    {"input": "What documents must be carried on board a vessel engaged in international trade?", "category": "DEEP_RAG"},
    {"input": "Describe the day shapes for a vessel engaged in fishing.", "category": "DEEP_RAG"},
    {"input": "What is the required sound signal for a vessel in fog?", "category": "DEEP_RAG"},
    {"input": "Explain the meaning of a 'Safe Water' mark.", "category": "DEEP_RAG"},
    {"input": "What are the USCG requirements for visual distress signals?", "category": "DEEP_RAG"},
    {"input": "Describe the pilotage rules for entering a major US port.", "category": "DEEP_RAG"},
    {"input": "What is the penalty for violating maritime safety regulations?", "category": "DEEP_RAG"},
    {"input": "Explain the concept of 'Innocent Passage' in maritime law.", "category": "DEEP_RAG"},
    {"input": "What is the definition of a 'vessel' according to the USCG?", "category": "DEEP_RAG"}
]

# --- Candidates ---

class GemmaRouter:
    """Salesforce VoiceAgentRAG Style: 4B LLM as a router."""
    def __init__(self, uri="http://localhost:8001/v1/chat/completions"):
        self.uri = uri

    def __call__(self, text: str) -> str:
        prompt = f"""[ROUTING TASK]
Classify the user input into EXACTLY ONE category:
- IGNORE: Ambient chatter between passengers NOT directed at the agent (e.g. "Hand me the towel", "The water is blue").
- FAST_CHAT: Greetings, jokes, or casual questions directed TO QUINT (e.g. "Quint, how are you?", "Tell me a story").
- TOOL_CALL: Imperative commands for hardware status or navigation (e.g. "status report", "set waypoint").
- DEEP_RAG: Technical/legal questions requiring document retrieval (e.g. "USCG life jacket rules").

EXAMPLES:
Input: "Hey Sarah, check out that dolphin!" -> Category: IGNORE
Input: "Quint, what do you think of that dolphin?" -> Category: FAST_CHAT
Input: "Is it lunch time yet?" -> Category: IGNORE
Input: "Quint, give me a status report." -> Category: TOOL_CALL
Input: "What are the rules for fire extinguishers on a 40ft boat?" -> Category: DEEP_RAG

Input: "{text}"
Category:"""
        
        try:
            resp = requests.post(self.uri, json={
                "model": "google/gemma-4-E4B-it",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "temperature": 0.0
            }, timeout=5.0)
            
            if resp.status_code == 200:
                result = resp.json()['choices'][0]['message']['content'].strip().upper()
                # Extract first word to be safe
                for cat in ["IGNORE", "FAST_CHAT", "TOOL_CALL", "DEEP_RAG"]:
                    if cat in result: return cat
            return "FAST_CHAT" # Fallback
        except Exception as e:
            return "ERROR"

def evaluate_router(name, router_fn):
    print(f"\n--- 🧪 Router Gym: {name} ---")
    correct = 0
    total = len(DATASET)
    latencies = []

    for item in DATASET:
        start = time.perf_counter()
        predicted = router_fn(item["input"])
        latencies.append(time.perf_counter() - start)
        
        if predicted == item["category"]:
            correct += 1
        else:
            print(f"❌ Mismatch: '{item['input'][:30]}...' | Expected: {item['category']} | Got: {predicted}")

    accuracy = (correct / total) * 100
    avg_latency = (sum(latencies) / total) * 1000
    
    print(f"\n✅ Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print(f"⏱️ Avg Latency: {avg_latency:.2f}ms")
    print(f"-----------------------------------------")

if __name__ == "__main__":
    # Baseline: Simple Keyword Mock
    def mock_router(text):
        text_l = text.lower()
        if any(x in text_l for x in ["pass", "look at", "sarah", "lunch", "sunscreen", "rope"]): return "IGNORE"
        if any(x in text_l for x in ["requirement", "rule", "law", "colreg", "meaning of"]): return "DEEP_RAG"
        if any(x in text_l for x in ["status", "telemetry", "waypoint", "vram", "cpu", "mark"]): return "TOOL_CALL"
        return "FAST_CHAT"

    evaluate_router("Keyword Baseline (Hardened)", mock_router)
    
    # Candidate: VoiceAgentRAG (Gemma-4B)
    evaluate_router("VoiceAgentRAG (Gemma-3-4B)", GemmaRouter())
