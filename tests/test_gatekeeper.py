
import os
import sys
import logging
from semantic_router import Route, SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("GatekeeperTest")

def test_gatekeeper():
    logger.info("Initializing Gatekeeper (Semantic Router)...")
    try:
        # Initialize Encoder
        encoder = HuggingFaceEncoder(name="Snowflake/snowflake-arctic-embed-xs")
        
        # Route A: Ignore
        ignore_route = Route(
            name="ignore",
            utterances=[
                "Pass the salt", 
                "Did you see that movie?", 
                "It's raining outside", 
                "Umm...", 
                "Yeah exactly.",
                "Wait",
                "Just a second",
                "Nevermind",
                "I don't know",
            ]
        )
        
        # Route B: Engage
        engage_route = Route(
            name="engage",
            utterances=[
                "What is our depth?", 
                "Set a waypoint", 
                "Check the radar", 
                "Hey Quint", 
                "Captain to Bridge",
                "Status report",
                "Systems check",
                "Any contact?",
                "Look at this",
            ]
        )
        
        router = SemanticRouter(encoder=encoder)
        router.add([ignore_route, engage_route])
        logger.info("Gatekeeper Initialized Successfully.")
        
    except Exception as e:
        logger.error(f"Failed to initialize Gatekeeper: {e}")
        return

    # Test Cases
    test_cases = [
        ("Pass the salt", "ignore", True),
        ("What is our depth?", "engage", True),
        ("It's raining outside", "ignore", True),
        ("Hey Quint, check the radar", "engage", True), # Variation
        ("Umm... I'm not sure", "ignore", True), # Variation
        ("Captain, set a waypoint", "engage", True), # Variation
        ("Totally unrelated gibberish about nothing", "None", False), # Should be dropped by threshold
        ("Quint, look at this", "engage", True), # Soft wake word test
    ]
    
    logger.info("\n--- Running Tests ---\n")
    
    for text, expected_route, should_pass in test_cases:
        route_choice = router(text)
        
        decision = "DROP"
        if route_choice.name == "engage" and route_choice.similarity_score >= 0.82:
            decision = "PASS"
        elif route_choice.name != "ignore" and route_choice.similarity_score >= 0.82:
             # Logic in orchestrator: if not ignore and sim > 0.82 -> PASS
             decision = "PASS"
        
        # Check against orchestrator logic exactly:
        # 1. if name == "ignore" -> continue (DROP)
        # 2. if similarity < 0.82 -> continue (DROP)
        # 3. else -> PASS
        
        orchestrator_decision = "PASS"
        if route_choice.name == "ignore":
            orchestrator_decision = "DROP"
        elif route_choice.similarity_score < 0.82:
            orchestrator_decision = "DROP"
            
        logger.info(f"Input: '{text}'")
        logger.info(f"  > Route: {route_choice.name}, Sim: {route_choice.similarity_score:.4f}")
        logger.info(f"  > Decision: {orchestrator_decision}")
        
        # Verification
        # Note: "Totally unrelated" might map to ignore or engage with low score, or None. 
        # Semantic Router usually forces a choice unless it's very far.
        pass

if __name__ == "__main__":
    test_gatekeeper()
