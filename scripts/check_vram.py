#!/usr/bin/env python3
import subprocess
import sys
import csv
import io
import os
import re

# Constants
REQUIRED_VRAM_GB = 18.0
STACK_NAME = "chatty-buoy" 

def get_stack_pids():
    """Returns a set of PIDs belonging to the docker compose stack."""
    pids = set()
    try:
        cmd = ["docker", "compose", "ps", "-q"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        container_ids = result.stdout.strip().split()

        for cid in container_ids:
            if not cid: continue
            try:
                # Use docker top to find host PIDs
                top_cmd = ["docker", "top", cid]
                top_res = subprocess.run(top_cmd, capture_output=True, text=True)
                if top_res.returncode != 0: continue
                
                lines = top_res.stdout.strip().split('\n')
                if len(lines) > 1:
                    headers = lines[0].split()
                    try: pid_idx = headers.index("PID")
                    except ValueError: continue

                    for line in lines[1:]:
                        parts = line.split()
                        if len(parts) > pid_idx:
                            pids.add(parts[pid_idx])
            except Exception: continue
    except subprocess.CalledProcessError: pass 
    return pids

def get_gpu_processes_nvidia():
    """Returns nvidia-smi process info or None if unavailable."""
    processes = []
    total_mem = 0
    try:
        cmd = ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader,nounits"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0: return None
        
        reader = csv.reader(io.StringIO(result.stdout))
        for row in reader:
            if len(row) < 3: continue
            pid = row[0].strip()
            try: mem_mb = float(row[1].strip())
            except: mem_mb = 0.0
            name = row[2].strip()
            
            if mem_mb > 0:
                processes.append({'pid': pid, 'memory_mb': mem_mb, 'name': name})
                total_mem += mem_mb
    except: return None
    
    if total_mem == 0: return None # Heuristic: if total is 0, nvidia-smi likely not tracking
    return processes

def get_free_vram_nvidia():
    try:
        cmd = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        if lines:
            val = float(lines[0].strip())
            if val > 1.0: return val
    except: return 0.0
    return 0.0

def get_system_ram_free_mb():
    """Parse /proc/meminfo for MemAvailable."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    # Format: MemAvailable:    12345 kB
                    parts = line.split()
                    kb = int(parts[1])
                    return kb / 1024.0
    except: pass
    return 0.0

def get_process_rss_mb(pid):
    """Get RSS in MB for a PID using /proc/[pid]/status."""
    try:
        with open(f"/proc/{pid}/status", 'r') as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    kb = int(parts[1])
                    return kb / 1024.0
    except: pass
    return 0.0

def get_memory_fallback(stack_pids):
    """Fallback using standard 'ps' command or /proc scanning."""
    processes = []
    
    # 1. Check Stack Processes
    for pid in stack_pids:
        mem_mb = get_process_rss_mb(pid)
        if mem_mb > 0:
            # Try to get name
            try:
                with open(f"/proc/{pid}/comm", 'r') as f:
                    name = f.read().strip()
            except: name = "unknown"
            
            processes.append({'pid': str(pid), 'memory_mb': mem_mb, 'name': name})

    # 2. Check Top External Processes using `ps`
    try:
        # ps -e -o pid,rss,comm --sort=-rss | head -n 11 (header + 10)
        cmd = ["ps", "-e", "-o", "pid,rss,comm", "--sort=-rss"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        lines = result.stdout.strip().split('\n')
        count = 0
        for line in lines[1:]: # Skip header
            if count >= 10: break
            parts = line.split(maxsplit=2)
            if len(parts) < 3: continue
            
            pid = parts[0].strip()
            if pid in stack_pids: continue
            
            try: rss_kb = int(parts[1])
            except: continue
            
            mem_mb = rss_kb / 1024.0
            name = parts[2].strip()
            
            if mem_mb > 100: # Filter small noise
                processes.append({'pid': pid, 'memory_mb': mem_mb, 'name': name})
                count += 1
    except: pass
    
    return processes

def main():
    print("🔍 Analyzing Memory Usage (Unified/VRAM)...")
    stack_pids = get_stack_pids()
    
    # Try Nvidia
    gpu_procs = get_gpu_processes_nvidia()
    free_mb = get_free_vram_nvidia()
    source = "NVIDIA-SMI"
    
    if gpu_procs is None or free_mb <= 1.0:
        source = "SYSTEM RAM (Unified)"
        gpu_procs = get_memory_fallback(stack_pids)
        free_mb = get_system_ram_free_mb()
        
    stack_usage_mb = 0.0
    external_usage_mb = 0.0
    
    print(f"\nSOURCE: {source}")
    print(f"{'PID':<8} | {'MEM (MB)':<10} | {'TYPE':<10} | {'PROCESS'}")
    print("-" * 60)
    
    seen_pids = set()
    for proc in gpu_procs:
        pid = proc['pid']
        if pid in seen_pids: continue
        seen_pids.add(pid)
        
        mem = proc['memory_mb']
        name = proc['name']
        is_stack = pid in stack_pids
        p_type = "[STACK]" if is_stack else "[EXTERNAL]"
        
        if is_stack: stack_usage_mb += mem
        else: external_usage_mb += mem
        
        if len(name) > 30: name = name[:27] + "..."
        print(f"{pid:<8} | {mem:<10.0f} | {p_type:<10} | {name}")
        
    print("-" * 60)
    
    total_available_mb = free_mb + stack_usage_mb
    total_available_gb = total_available_mb / 1024.0
    
    print(f"\n📊 Summary:")
    print(f"   Free Memory:         {free_mb/1024.0:.2f} GB")
    print(f"   Stack Reclaimable:   {stack_usage_mb/1024.0:.2f} GB")
    print(f"   External Used:       {external_usage_mb/1024.0:.2f} GB")
    print(f"   ---------------------------")
    print(f"   Total Available:     {total_available_gb:.2f} GB")
    print(f"   Required:            {REQUIRED_VRAM_GB:.2f} GB")
    
    if total_available_gb < REQUIRED_VRAM_GB:
        print("\n❌ INSUFFICIENT MEMORY DETECTED")
        print(f"   The stack requires at least {REQUIRED_VRAM_GB} GB.")
        print("   External processes are consuming too much memory.")
        print("   Please kill highlighted external PIDs or run 'sudo drop_caches'.")
        sys.exit(1)
    else:
        print(f"\n✅ Memory Check Passed ({total_available_gb:.1f} GB Available). Proceeding...")
        sys.exit(0)

if __name__ == "__main__":
    main()
