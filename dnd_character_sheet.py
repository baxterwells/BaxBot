from typing import List, Dict, Any

class Weapon:
    """Represents a detailed weapon object within a character sheet."""
    def __init__(self, name: str, damage_die: str, damage_type: str, attack_bonus: int = 0):
        self.name = name
        self.damage_die = damage_die  # e.g., "1d8"
        self.damage_type = damage_type  # e.g., "Slashing"
        self.attack_bonus = attack_bonus

    def __repr__(self):
        return f"{self.name} ({self.damage_die} {self.damage_type} + {self.attack_bonus})"

    def to_dict(self):
        return vars(self)


class DnDCharacterSheet:
    def __init__(self, name: str, race: str, char_class: str, backstory: str):
        # Core Identity
        self.name = name
        self.race = race
        self.char_class = char_class
        self.backstory = backstory
        self.level = 1

        # Primary Attributes (Raw Scores)
        self.attributes = {
            "STR": 10,
            "DEX": 10,
            "CON": 10,
            "INT": 10,
            "WIS": 10,
            "CHA": 10
        }

        # Combat Stats
        self.hp_current = 10
        self.hp_max = 10
        self.speed = 30
        self.initiative_bonus = 0
        
        # Inventory/Equipment
        self.weapons: List[Weapon] = []

    def _calculate_modifier(self, score: int) -> int:
        """Standard D&D 5e modifier formula: (score - 10) / 2, rounded down."""
        return (score - 10) // 2

    def get_modifier(self, stat_name: str) -> int:
        """Returns the calculated modifier for a specific stat."""
        score = self.attributes.get(stat_name.upper(), 10)
        return self._calculate_modifier(score)

    def update_attribute(self, stat_name: str, new_score: int):
        """Updates a raw attribute score."""
        stat_name = stat_name.upper()
        if stat_name in self.attributes:
            self.attributes[stat_name] = new_score
            # Update initiative bonus automatically if DEX changes
            if stat_name == "DEX":
                self.initiative_bonus = self.get_modifier("DEX")
        else:
            print(f"Error: {stat_name} is not a valid attribute.")

    def update_hp(self, amount: int):
        """Adjusts current HP. Positive for healing, negative for damage."""
        self.hp_current += amount
        # Clamp HP between 0 and Max
        if self.hp_current > self.hp_max:
            self.hp_current = self.hp_max
        elif self.hp_current < 0:
            self.hp_current = 0

    def set_hp_max(self, new_max: int):
        """Sets the maximum HP (useful for leveling up)."""
        self.hp_max = new_max
        if self.hp_current > self.hp_max:
            self.hp_current = self.hp_max

    def add_weapon(self, weapon: Weapon):
        """Adds a weapon object to the character's arsenal."""
        self.weapons.append(weapon)

    def remove_weapon(self, weapon_name: str):
        """Removes a weapon by its name."""
        self.weapons = [w for w in self.weapons if w.name != weapon_name]

    def update_backstory(self, new_text: str):
        """Updates the narrative history."""
        self.backstory = new_text

    def get_full_sheet(self) -> Dict[str, Any]:
        """Returns a complete dictionary representation of the character for BaxBot to read."""
        return {
            "name": self.name,
            "race": self.race,
            "class": self.char_class,
            "level": self.level,
            "backstory": self.backstory,
            "stats": {
                stat: {
                    "score": score,
                    "modifier": self._calculate_modifier(score)
                } for stat, score in self.attributes.items()
            },
            "combat": {
                "hp": f"{self.hp_current}/{self.hp_max}",
                "speed": self.speed,
                "initiative_bonus": self.initiative_bonus
            },
            "weapons": [w.to_dict() for w in self.weapons]
        }

# --- TEST/EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. Create the Character
    hero = DnDCharacterSheet(
        name="Kaelen", 
        race="Elf", 
        char_class="Ranger", 
        backstory="A wanderer of the Silver Woods, seeking the lost ember."
    )

    # 2. Set Attributes (Surgical Updates)
    hero.update_attribute("DEX", 16)  # Should give +3 modifier
    hero.update_attribute("STR", 12)  # Should give +1 modifier
    hero.update_attribute("CON", 14)  # Should give +2 modifier

    # 3. Setup HP and Speed
    hero.set_hp_max(25)
    hero.update_hp(15) # Set current HP to 15
    hero.speed = 35

    # 4. Add a Weapon Object
    longbow = Weapon("Silverleaf Longbow", "1d8", "Piercing", attack_bonus=3)
    hero.add_weapon(longbow)

    # 5. Print the result for verification
    import pprint
    print("--- CHARACTER SHEET ---")
    pprint.pprint(hero.get_full_sheet())
