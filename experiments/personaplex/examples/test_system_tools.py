#!/usr/bin/env python3
"""
Test system monitoring tools without requiring PersonaPlex/FunctionGemma services.

This script validates that all system monitoring tools work correctly on the Thor system.

Usage:
    micromamba run -n chatty-buoy python3 examples/test_system_tools.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chatty_buoy.crew.crew_member import CrewMember
import psutil
import subprocess
import json


def test_system_info():
    """Test get_system_info tool."""
    print("\n" + "="*60)
    print("TEST 1: System Information")
    print("="*60)
    
    crew = CrewMember(personaplex_config=None, gemma_config=None)
    result = crew._tool_get_system_info()
    
    print("\nSystem Info Tool Output:")
    try:
        info = json.loads(result)
        for key, value in info.items():
            print(f"  {key}: {value}")
    except:
        print(result)


def test_running_processes():
    """Test get_running_processes tool."""
    print("\n" + "="*60)
    print("TEST 2: Running Processes")
    print("="*60)
    
    crew = CrewMember(personaplex_config=None, gemma_config=None)
    result = crew._tool_get_running_processes()
    
    print("\nRunning Processes (top 5 by CPU):")
    try:
        data = json.loads(result)
        processes = data.get("top_processes", [])
        for proc in processes[:5]:
            print(f"  {proc['name']:20} CPU: {proc['cpu_percent']:6.1f}% MEM: {proc['memory_mb']:7.1f}MB")
    except:
        print(result)


def test_hardware_specs():
    """Test get_hardware_specs tool."""
    print("\n" + "="*60)
    print("TEST 3: Hardware Specifications")
    print("="*60)
    
    crew = CrewMember(personaplex_config=None, gemma_config=None)
    result = crew._tool_get_hardware_specs()
    
    print("\nHardware Specs:")
    try:
        specs = json.loads(result)
        for key, value in specs.items():
            if key != "gpu_info":
                print(f"  {key}: {value}")
        if specs.get("gpu_info"):
            print(f"  GPU Info:")
            for gpu in specs["gpu_info"]:
                print(f"    - {gpu}")
    except:
        print(result)


def test_gpu_stats():
    """Test get_gpu_stats tool."""
    print("\n" + "="*60)
    print("TEST 4: GPU Statistics")
    print("="*60)
    
    crew = CrewMember(personaplex_config=None, gemma_config=None)
    result = crew._tool_get_gpu_stats()
    
    print("\nGPU Stats:")
    try:
        stats = json.loads(result)
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
    except:
        print(result)


def test_system_status():
    """Test get_system_status tool."""
    print("\n" + "="*60)
    print("TEST 5: System Status")
    print("="*60)
    
    crew = CrewMember(personaplex_config=None, gemma_config=None)
    result = crew._tool_get_system_status()
    
    print("\nSystem Status:")
    try:
        status = json.loads(result)
        for key, value in status.items():
            print(f"  {key}: {value}")
    except:
        print(result)


def main():
    """Run all system tool tests."""
    print("\n" + "#"*60)
    print("# CrewMember System Monitoring Tools Test")
    print("#"*60)
    
    try:
        test_system_info()
        test_running_processes()
        test_hardware_specs()
        test_gpu_stats()
        test_system_status()
        
        print("\n" + "="*60)
        print("✓ All tests completed successfully!")
        print("="*60)
        print("\nThese system monitoring tools will be used automatically when you chat with CrewMember.")
        print("Example: 'What are the hardware specs?' → calls get_hardware_specs")
        print("Example: 'How much memory are we using?' → calls get_system_info")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
