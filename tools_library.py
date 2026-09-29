import subprocess
import json # Added for cleaner output handling
from abc import ABC, abstractmethod
from memory_manager import PersonalAgentMemory

# Import the new classes
from dnd_character_sheet import DnDCharacterSheet, Weapon

class BaseTool(ABC):
    """The abstract blueprint for every tool BaxBot can use."""
    @abstractmethod
    def execute(self, args: list = None) -> str:
        pass

class FamilyPhotoSorter(BaseTool):
    def execute(self, args: list = None) -> str:
        print("[FamilyPhotoSorter]: Running Family Photo Sorter...")
        result = subprocess.run(["python3", "family_photo_sorter.py"], capture_output=True, text=True)
        return result.stdout if result.stdout else "Process complete (no output)."

class SystemStatsTool(BaseTool):
    """A new tool to show off your M4 Max capabilities."""
    def execute(self, args: list = None) -> str:
        print("[SystemStatsTool]: Gathering system statistics...")
        cmd = "top -l 1 | head -n 10" 
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout

class MemoryManagerTool(BaseTool):
    def __init__(self):
        self.manager = PersonalAgentMemory()

    def execute(self, args: list = None) -> str:
        if not args:
            return self.manager.ingest_vault("memory_vault")

        if len(args) == 3:
            category, key, content = args[0], args[1], args[2]
            return self.manager.update_entry(category, key, content)
        
        return "Error: Memory update requires category, key, and content."

class DnDCharacterSheetTool(BaseTool):
    """Tool to manage the current active DnD character."""
    def __init__(self):
        # In a production version, you'd likely load this from a JSON file
        # For now, we initialize a default character
        self.character = DnDCharacterSheet(
            name="Kaelen", 
            race="Elf", 
            char_class="Ranger", 
            backstory="A wanderer of the Silver Woods."
        )

    def execute(self, args: list = None) -> str:
        if not args:
            return "Error: DnD tool requires a command (get_sheet, update_attr, update_hp, add_weapon)."

        command = args[0]

        try:
            # COMMAND: get_sheet
            if command == "get_sheet":
                return json.dumps(self.character.get_full_sheet(), indent=2)

            # COMMAND: update_attr | [stat_name] | [new_score]
            # Example: CALL_TOOL: dnd_character_sheet | update_attr | STR | 18
            elif command == "update_attr" and len(args) == 3:
                stat_name = args[1]
                new_score = int(args[2])
                self.character.update_attribute(stat_name, new_score)
                return f"{stat_name} updated to {new_score}."

            # COMMAND: update_hp | [amount]
            # Example: CALL_TOOL: dnd_character_sheet | update_hp | -5
            elif command == "update_hp" and len(args) == 2:
                amount = int(args[1])
                self.character.update_hp(amount)
                return f"HP adjusted by {amount}. Current: {self.character.hp_current}/{self.character.hp_max}"

            # COMMAND: add_weapon | [name] | [die] | [type] | [bonus]
            # Example: CALL_TOOL: dnd_character_sheet | add_weapon | Longbow | 1d8 | Piercing | 3
            elif command == "add_weapon" and len(args) == 5:
                w_name, w_die, w_type, w_bonus = args[1], args[2], args[3], int(args[4])
                new_weapon = Weapon(w_name, w_die, w_type, w_bonus)
                self.character.add_weapon(new_weapon)
                return f"{w_name} added to inventory."

            else:
                return f"Error: Unknown command '{command}' or incorrect arguments."

        except Exception as e:
            return f"Error executing DnD tool: {str(e)}"
