#!/usr/bin/env python3
"""
PDF Downloader Script for Chatty Buoy Knowledge Base

This script allows users to download the latest versions of canonical nautical publications
such as the USCG Coast Pilot series, Navigation Rules, and other federal requirements.
It supports grouped selection and custom PDF URLs.
"""

import os
import json
import requests
import sys
from typing import Dict, List, Optional, Tuple, Any

# Try to import UI libraries
try:
    import questionary
except ImportError:
    questionary = None

try:
    from rich.console import Console
    from rich.progress import Progress, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn
    from rich.table import Table
    from rich import print as rprint
    console = Console()
except ImportError:
    console = None
    print("Warning: 'rich' library not found. Falling back to basic console output.")

# Constants
PDF_DIR = "pdfs"
CUSTOM_PDFS_FILE = "custom_pdfs.json"

# Default Resources
COAST_PILOT_BASE_URL = "https://nauticalcharts.noaa.gov/publications/coast-pilot/files"

DEFAULT_RESOURCES = {
    "Coast Pilot Series": {
        f"Coast Pilot {i}": (f"{COAST_PILOT_BASE_URL}/cp{i}/CPB{i}_WEB.pdf", f"USCG_Coast_Pilot_{i}.pdf") for i in range(1, 11)
    },
    "Regulations & Guides": {
        "Navigation Rules (International & Inland)": ("https://www.navcen.uscg.gov/sites/default/files/pdf/navRules/navrules.pdf", "USCG_Navigation_Rules.pdf"),
        "U.S. Aids to Navigation System": ("https://www.uscgboating.org/images/486.PDF", "USCG_Aids_to_Navigation_System.pdf"),
        "Federal Requirements for Recreational Boats": ("https://www.uscgboating.org/assets/1/AssetManager/Boaters-Guide-to-Federal-Requirements-for-Recreational-Boats.pdf", "USCG_Boaters_Guide_to_Federal_Requirements.pdf"),
        "Nautical Knowledge (SPC)": ("https://coastfish.spc.int/Sections/training/fts_pdf/statutory/nautical_lg_en.pdf", "Secretariat of the Pacific - Nautical Knowledge for Pacific Island Mariners.pdf")
    }
}


def ensure_pdf_dir():
    """Ensures the PDF directory exists."""
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
        if console:
            console.print(f"[green]Created directory:[/green] {PDF_DIR}")
        else:
            print(f"Created directory: {PDF_DIR}")


def load_custom_pdfs() -> Dict[str, str]:
    """Loads custom PDF URLs from the local JSON file."""
    if os.path.exists(CUSTOM_PDFS_FILE):
        try:
            with open(CUSTOM_PDFS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            msg = f"Warning: Could not decode {CUSTOM_PDFS_FILE}. Starting with empty custom list."
            if console:
                console.print(f"[yellow]{msg}[/yellow]")
            else:
                print(msg)
    return {}


def save_custom_pdfs(custom_pdfs: Dict[str, str]):
    """Saves custom PDF URLs to the local JSON file."""
    with open(CUSTOM_PDFS_FILE, 'w') as f:
        json.dump(custom_pdfs, f, indent=4)
    if console:
        console.print(f"[green]Saved custom URL to[/green] {CUSTOM_PDFS_FILE}")
    else:
        print(f"Saved custom URL to {CUSTOM_PDFS_FILE}")


def download_file(url: str, filename: str):
    """Downloads a file from a URL to the PDF directory with a progress indicator."""
    ensure_pdf_dir()
    filepath = os.path.join(PDF_DIR, filename)

    if console:
        # Rich progress bar
        try:
            with Progress(
                TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
                BarColumn(bar_width=None),
                "[progress.percentage]{task.percentage:>3.1f}%",
                "•",
                DownloadColumn(),
                "•",
                TransferSpeedColumn(),
                transient=True,
            ) as progress:
                task_id = progress.add_task("download", filename=filename, start=False)
                response = requests.get(url, stream=True)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                progress.update(task_id, total=total_size)
                progress.start_task(task_id)
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        progress.update(task_id, advance=len(chunk))
            console.print(f"[green]✔ Downloaded {filename}[/green]")
            return
        except Exception as e:
            console.print(f"[red]Error downloading {url}: {e}[/red]")
            return

    # Basic fallback
    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = int(50 * downloaded / total_size)
                        sys.stdout.write(f"\r[{'=' * percent}{' ' * (50 - percent)}] {downloaded}/{total_size} bytes")
                        sys.stdout.flush()
        print("\nDownload complete!")
    except requests.exceptions.RequestException as e:
        print(f"\nError downloading {url}: {e}")


def get_all_resources() -> Dict[str, Dict[str, str]]:
    """Combines default and custom resources."""
    resources = DEFAULT_RESOURCES.copy()
    custom = load_custom_pdfs()
    if custom:
        resources["Custom URLs"] = custom
    return resources


def add_custom_url():
    """Prompts user to add a custom URL."""
    if questionary:
        name = questionary.text("Enter a name for the PDF:").ask()
        url = questionary.text("Enter the URL:").ask()
    else:
        name = input("Enter a name for the PDF: ").strip()
        url = input("Enter the URL: ").strip()
        
    if name and url:
        custom_pdfs = load_custom_pdfs()
        custom_pdfs[name] = url
        save_custom_pdfs(custom_pdfs)
        if console:
            console.print(f"[green]Added '{name}' to custom list.[/green]")
        else:
            print(f"Added '{name}' to custom list.")


def select_and_download(category_name: str, items: Dict[str, str]):
    """Handles selection of items within a category and downloads them."""
    if not items:
        if console:
             console.print("[yellow]No items in this category.[/yellow]")
        else:
             print("No items in this category.")
        return

    selected_items = []
    
    if questionary:
        choices = list(items.keys())
        selection = questionary.checkbox(
            f"Select {category_name} to download:",
            choices=choices
        ).ask()
        if selection:
            selected_items = selection
    else:
        # Fallback text-based selection
        item_list = list(items.keys())
        if console:
            table = Table(title=f"Select {category_name}")
            table.add_column("ID", justify="right", style="cyan", no_wrap=True)
            table.add_column("Name", style="magenta")
            for idx, name in enumerate(item_list, 1):
                table.add_row(str(idx), name)
            console.print(table)
            console.print("Enter IDs to download (e.g. '1, 3, 5-7' or 'all'):")
        else:
            print(f"\nSelect {category_name}:")
            for idx, name in enumerate(item_list, 1):
                print(f"{idx}. {name}")
            print("Enter IDs to download (e.g. '1, 3, 5-7' or 'all'):")

        user_input = input("> ").strip().lower()
        if user_input == 'all':
            selected_items = item_list
        elif user_input:
            # Parse input
            parts = user_input.replace(',', ' ').split()
            for part in parts:
                if '-' in part:
                    try:
                        start, end = map(int, part.split('-'))
                        for i in range(start, end + 1):
                            if 1 <= i <= len(item_list):
                                selected_items.append(item_list[i-1])
                    except ValueError:
                        pass
                elif part.isdigit():
                    idx = int(part)
                    if 1 <= idx <= len(item_list):
                        selected_items.append(item_list[idx-1])

    if not selected_items:
        if console:
            console.print("[yellow]No items selected.[/yellow]")
        else:
            print("No items selected.")
        return

    # Download selected
    for name in selected_items:
        item = items[name]
        if isinstance(item, tuple):
            url, filename = item
        else:
            url = item
            filename = url.split('/')[-1]
            if '?' in filename:
                filename = filename.split('?')[0]
            if not filename.lower().endswith('.pdf'):
                filename += ".pdf"
            
        download_file(url, filename)


def main():
    while True:
        resources = get_all_resources()
        categories = list(resources.keys())
        
        # Add special actions
        ACTIONS = ["Add Custom URL", "Exit"]
        options = categories + ACTIONS
        
        selected_option = None
        
        if questionary:
            selected_option = questionary.select(
                "Select a category or action:",
                choices=options
            ).ask()
        else:
            if console:
                console.print("\n[bold]Main Menu[/bold]")
                for idx, opt in enumerate(options, 1):
                    console.print(f"[cyan]{idx}[/cyan]. {opt}")
                choice = input("\nEnter choice: ").strip()
            else:
                print("\nMain Menu")
                for idx, opt in enumerate(options, 1):
                    print(f"{idx}. {opt}")
                choice = input("\nEnter choice: ").strip()
                
            if choice.isdigit() and 1 <= int(choice) <= len(options):
                selected_option = options[int(choice)-1]
        
        if not selected_option:
            continue
            
        if selected_option == "Exit":
            if console:
                console.print("[bold blue]Goodbye![/bold blue]")
            else:
                print("Goodbye!")
            break
        elif selected_option == "Add Custom URL":
            add_custom_url()
        elif selected_option in resources:
            select_and_download(selected_option, resources[selected_option])

if __name__ == "__main__":
    main()
