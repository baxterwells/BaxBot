import subprocess
import ollama
import os
from rich.console import Console
import json 
from abc import ABC, abstractmethod
from memory_manager import PersonalAgentMemory

class BaseTool(ABC):
    console = Console()
    """The abstract blueprint for every tool that BaxBot can use."""
    @abstractmethod
    def execute(self, args: dict = None) -> str:
        pass

class PhotoSorter(BaseTool):
    def execute(self, args: dict = None) -> str:
        self.console.print("\n[bold][ PhotoSorter][/bold]: Running Photo Sorrer...")
        result = subprocess.run(["python3", "photo_sorter.py"], capture_output=True, text=True)
        return result.stdout if result.stdout else "Process complete (no output)."

class SystemStatsTool(BaseTool):
    """A new tool to show off your M4 Max capabilities."""
    def execute(self, args: dict = None) -> str:
        self.console.print("\n[bold][󰣖 SystemStats][/bold]: Gathering system statistics...")
        cmd = "top -l 1 | head -n 10" 
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return "\n" + result.stdout

class MemoryManagerTool(BaseTool):
    def __init__(self):
        self.manager = PersonalAgentMemory()

    def execute(self, args: dict = None) -> str:
        # If args is None or an empty dict, perform the full vault ingestion/refresh
        if not args:
            return self.manager.ingest_vault("memory_vault")

        # Extract values using the keys defined in the system prompt
        category = args.get("category")
        key = args.get("key")
        content = args.get("content")

        if category and key and content:
            return self.manager.update_entry(category, key, content)
        
        return "Error: Memory update requires 'category', 'key', and 'content'."

class DnDExpertTool(BaseTool):
    """A specialized sub-session tool for D&D Rules and Mechanics."""
    
    def execute(self, args: dict = None) -> str:
        # Determine which model to use via the "model" key in the dictionary
        if isinstance(args, dict):
            model_name = args.get("model", "gemma2:27b")
        else:
            model_name = "gemma2:27b" 
        
        # Find the path to the dnd_session script
        script_path = os.path.join(os.path.dirname(__file__), "dnd_session.py")

        try:
            # THE HANDOVER
            result = subprocess.run(["python3", script_path, model_name])
            
            if result.returncode == 0:
                return "User exited D&D Mode."
            else:
                return "D&D Session ended with an error (non-zero exit code)."
                
        except Exception as e:
            self.console.print(f"[bold red]Failed to launch D&D Session: {e}[/bold red]")
            return f"Error launching tool: {e}"
