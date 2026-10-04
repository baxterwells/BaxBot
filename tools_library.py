import subprocess
import ollama
import os
from rich.console import Console
import json # Added for cleaner output handling
from abc import ABC, abstractmethod
from memory_manager import PersonalAgentMemory

class BaseTool(ABC):
    console = Console()
    """The abstract blueprint for every tool BaxBot can use."""
    @abstractmethod
    def execute(self, args: list = None) -> str:
        pass

class PhotoSorter(BaseTool):
    def execute(self, args: list = None) -> str:
        self.console.print("\n[bold][ PhotoSorter][/bold]: Running Photo Sorter...")
        result = subprocess.run(["python3", "photo_sorter.py"], capture_output=True, text=True)
        return result.stdout if result.stdout else "Process complete (no output)."

class SystemStatsTool(BaseTool):
    """A new tool to show off your M4 Max capabilities."""
    def execute(self, args: list = None) -> str:
        self.console.print("\n[bold][󰣖 SystemStats][/bold]: Gathering system statistics...")
        cmd = "top -l 1 | head -n 10" 
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return "\n" + result.stdout

class MemoryManagerTool(BaseTool):
    def __init__(self):
        self.manager = PersonalAgentMemory()

    def execute(self, args: list = None) -> str:
        #print("ARGS: ", args)
        if not args:
            return self.manager.ingest_vault("memory_vault")

        if len(args) == 3:
            category, key, content = args[0], args[1], args[2]
            return self.manager.update_entry(category, key, content)
        
        return "Error: Memory update requires category, key, and content."

class DnDExpertTool(BaseTool):
    """A specialized sub-session tool for D&D Rules and Mechanics."""
    
    def execute(self, args: dict = None) -> str: # Changed signature from list to dict
        # 1. Determine which model to use via the "model" key in the dictionary
        # This is much safer than trying to index into a list with [0]
        if isinstance(args, dict):
            model_name = args.get("model", "gemma2:27b")
        else:
            model_name = "gemma2:27b" 
        
        # 2. Find the path to the dnd_session script
        # os.path.dirname(__file__) ensures we find the script in the same folder as this tool
        script_path = os.path.join(os.path.dirname(__file__), "dnd_session.py")

        try:
            # 3. THE HANDOVER
            # subprocess.run 'freezes' the BaxBot process and starts the new one.
            # This allows the new process to have full control of the terminal.
            # We pass the model_name as a command line argument to the script.
            result = subprocess.run(["python3", script_path, model_name])
            
            # Check if the sub-process finished successfully
            if result.returncode == 0:
                return "User exited D&D Mode."
            else:
                return "D&D Session ended with an error (non-zero exit code)."
                
        except Exception as e:
            self.console.print(f"[bold red]Failed to launch D&D Session: {e}[/bold red]")
            return f"Error launching tool: {e}"
