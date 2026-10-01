import subprocess
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

class FamilyPhotoSorter(BaseTool):
    def execute(self, args: list = None) -> str:
        self.console.print("[ FamilyPhotoSorter]: Running Family Photo Sorter...")
        result = subprocess.run(["python3", "family_photo_sorter.py"], capture_output=True, text=True)
        return result.stdout if result.stdout else "Process complete (no output)."

class SystemStatsTool(BaseTool):
    """A new tool to show off your M4 Max capabilities."""
    def execute(self, args: list = None) -> str:
        self.console.print("[󰣖 SystemStatsTool]: Gathering system statistics...")
        cmd = "top -l 1 | head -n 10" 
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout

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