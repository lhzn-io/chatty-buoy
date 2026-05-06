import time
import os
import json
import redis
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_STREAM_KEY = "vision_events"

console = Console()

def get_events(r_client, count=10):
    try:
        # Read the last N events from the stream
        # XREVRANGE key + - COUNT N
        events = r_client.xrevrange(REDIS_STREAM_KEY, max='+', min='-', count=count)
        return events
    except Exception as e:
        return []

def generate_layout(r_client):
    events = get_events(r_client, count=5)
    
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Timestamp", style="dim", width=20)
    table.add_column("Class / Type", width=15)
    table.add_column("Details / Reasoning", style="green")

    if not events:
        table.add_row("N/A", "N/A", "Waiting for vision events to appear in Redis...")
    else:
        for event_id, event_data in events:
            # Redis stream data is returned as bytes or strings in a dict
            ts_str = event_data.get('timestamp', '0')
            try:
                local_ts = datetime.fromtimestamp(float(ts_str)).strftime('%Y-%m-%d %H:%M:%S')
            except:
                local_ts = event_id.split('-')[0] # fallback to redis ID timestamp
                
            e_class = event_data.get('class', 'unknown')
            
            if e_class == 'scene_summary':
                details = event_data.get('content', 'No content')
            else:
                bearing = event_data.get('bearing', '?')
                range_est = event_data.get('range', '?')
                threat = event_data.get('threat', '?')
                details = f"Bearing: {bearing}°, Range: {range_est}m, Status: {threat}"
            
            table.add_row(local_ts, e_class, details)

    layout = Layout()
    layout.split_column(
        Layout(Panel(Text("Sentinel Vision Dashboard", justify="center", style="bold cyan")), size=3),
        Layout(table)
    )
    return layout

def main():
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.ping()
    except Exception as e:
        console.print(f"[bold red]Failed to connect to Redis at {REDIS_HOST}:{REDIS_PORT}[/bold red]")
        return

    with Live(generate_layout(r), refresh_per_second=2, console=console) as live:
        try:
            while True:
                time.sleep(2)
                live.update(generate_layout(r))
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()
