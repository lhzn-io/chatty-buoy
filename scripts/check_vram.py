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
GHOST_MEMORY_THRESHOLD_GB = 5.0 # Threshold to warn about unaccounted memory

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
        # Check if nvidia-smi works and reports memory
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
    
    # If nvidia-smi reports no processes but we are on an NVIDIA system, 
    # it might just be idle or using unified memory.
    return processes

def get_free_vram_nvidia():
    try:
        cmd = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        if lines and "[N/A]" not in lines[0]:
            val = float(lines[0].strip())
            if val > 1.0: return val
    except: pass
    return 0.0

def get_jtop_stats():
    """Returns memory stats from jtop if available."""
    try:
        from jtop import jtop
        with jtop() as jetson:
            if jetson.ok():
                mem = jetson.memory
                # jtop memory['RAM'] fields: tot, used, free, buffers, cached
                # 'used' here usually means total - free - buffers - cached
                return {
                    'total_mb': mem['RAM']['tot'] / 1024.0,
                    'used_mb': mem['RAM']['used'] / 1024.0,
                    'free_mb': mem['RAM']['free'] / 1024.0,
                    'cached_mb': mem['RAM']['cached'] / 1024.0,
                    'available_mb': (mem['RAM']['free'] + mem['RAM']['cached']) / 1024.0
                }
    except ImportError:
        pass
    except Exception:
        pass
    return None

def get_meminfo_stats():
    """Parse /proc/meminfo for memory stats."""
    stats = {}
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                parts = line.split()
                if not parts: continue
                key = parts[0].strip(':')
                val_kb = int(parts[1])
                stats[key] = val_kb / 1024.0 # Convert to MB
    except: pass
    
    if 'MemTotal' in stats:
        # Linux MemAvailable is a good estimate of free memory
        available = stats.get('MemAvailable', stats.get('MemFree', 0) + stats.get('Cached', 0))
        return {
            'total_mb': stats['MemTotal'],
            'used_mb': stats['MemTotal'] - stats.get('MemFree', 0) - stats.get('Cached', 0) - stats.get('Buffers', 0),
            'free_mb': stats.get('MemFree', 0),
            'cached_mb': stats.get('Cached', 0),
            'available_mb': available
        }
    return None

def get_all_process_rss_mb():
    """Sum of RSS of all processes in the system."""
    try:
        cmd = ["ps", "-e", "-o", "rss"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        total_rss_kb = 0
        for line in lines[1:]: # Skip header
            try: total_rss_kb += int(line.strip())
            except: pass
        return total_rss_kb / 1024.0
    except:
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
        cmd = ["ps", "-e", "-o", "pid,rss,comm", "--sort=-rss"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        lines = result.stdout.strip().split('\n')
        count = 0
        for line in lines[1:]: # Skip header
            if count >= 15: break
            parts = line.split(maxsplit=2)
            if len(parts) < 3: continue
            
            pid = parts[0].strip()
            if pid in stack_pids: continue
            
            try: rss_kb = int(parts[1])
            except: continue
            
            mem_mb = rss_kb / 1024.0
            name = parts[2].strip()
            
            if mem_mb > 50: # Filter small noise
                processes.append({'pid': pid, 'memory_mb': mem_mb, 'name': name})
                count += 1
    except: pass
    
    return processes

def main():
    print("🔍 Analyzing Memory Usage (Unified/VRAM)...")
    stack_pids = get_stack_pids()
    
    # 1. Gather System Stats
    jtop_stats = get_jtop_stats()
    sys_stats = jtop_stats if jtop_stats else get_meminfo_stats()
    
    # 2. Gather GPU Stats (if nvidia-smi works)
    gpu_procs = get_gpu_processes_nvidia()
    nvidia_free_mb = get_free_vram_nvidia()
    
    source = "jtop (Jetson/Thor)" if jtop_stats else "NVIDIA-SMI" if nvidia_free_mb > 0 else "SYSTEM RAM (Unified)"
    
    # Use system available memory if nvidia-smi is not reporting it
    available_mb = nvidia_free_mb if nvidia_free_mb > 0 else sys_stats['available_mb']
    
    # 3. Calculate Ghost Memory (Unaccounted memory)
    total_rss_mb = get_all_process_rss_mb()
    # Ghost Memory = Total Used (non-cached) - Total Process RSS
    # This represents memory held by kernel drivers (like nvmap/CUDA) or zombie containers
    ghost_mem_mb = sys_stats['used_mb'] - total_rss_mb
    if ghost_mem_mb < 0: ghost_mem_mb = 0 # Safety
    
    # 4. Identify Processes
    if gpu_procs:
        procs = gpu_procs
    else:
        procs = get_memory_fallback(stack_pids)
        
    stack_usage_mb = 0.0
    external_usage_mb = 0.0
    
    print(f"\nSOURCE: {source}")
    print(f"{'PID':<8} | {'MEM (MB)':<10} | {'TYPE':<10} | {'PROCESS'}")
    print("-" * 60)
    
    seen_pids = set()
    for proc in procs:
        pid = str(proc['pid'])
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
    
    total_available_mb = available_mb + stack_usage_mb
    total_available_gb = total_available_mb / 1024.0
    
    print(f"\n📊 Summary:")
    print(f"   Available (Free+Cached): {sys_stats['available_mb']/1024.0:6.2f} GB")
    print(f"   System Used (Active):    {sys_stats['used_mb']/1024.0:6.2f} GB")
    print(f"   - Process RSS Sum:       {total_rss_mb/1024.0:6.2f} GB")
    print(f"   - Ghost Memory (Driver): {ghost_mem_mb/1024.0:6.2f} GB ⚠️" if ghost_mem_mb/1024.0 > GHOST_MEMORY_THRESHOLD_GB else f"   - Ghost Memory (Driver): {ghost_mem_mb/1024.0:6.2f} GB")
    print(f"   ---------------------------")
    print(f"   Stack Reclaimable:       {stack_usage_mb/1024.0:6.2f} GB")
    print(f"   Total Reclaimable:       {total_available_gb:6.2f} GB")
    print(f"   Required:                {REQUIRED_VRAM_GB:6.2f} GB")
    
    # Ghost memory detection logic
    is_ghost_problem = (ghost_mem_mb / 1024.0 > GHOST_MEMORY_THRESHOLD_GB) and (not stack_pids)
    
    if is_ghost_problem:
        print("\n💀 WARNING: HIGH GHOST MEMORY DETECTED")
        print(f"   Detected {ghost_mem_mb/1024.0:.1f} GB of memory held by drivers or zombie processes,")
        print("   even though no active stack containers were found.")
        print("   This often happens when Docker containers are stopped but didn't release VRAM.")
        print("   Action: Try 'drop-caches()' (sync && echo 3 | sudo tee /proc/sys/vm/drop_caches)")
        print("           or 'sudo systemctl restart docker' if memory isn't reclaimed.")

    if total_available_gb < REQUIRED_VRAM_GB:
        print("\n❌ INSUFFICIENT MEMORY DETECTED")
        print(f"   The stack requires at least {REQUIRED_VRAM_GB} GB.")
        if external_usage_mb > 1024:
            print(f"   External processes are consuming {external_usage_mb/1024.0:.1f} GB.")
        print("   Please kill highlighted external PIDs or reclaim ghost memory.")
        sys.exit(1)
    else:
        print(f"\n✅ Memory Check Passed ({total_available_gb:.1f} GB Available). Proceeding...")
        sys.exit(0)

if __name__ == "__main__":
    main()
