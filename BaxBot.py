import os
import sys
import ollama
import json  # <--- NEW: Essential for JSON parsing
import datetime
from memory_manager import PersonalAgentMemory 
from tools_library import PhotoSorter, SystemStatsTool, MemoryManagerTool
from rich.console import Console
from rich.markdown import Markdown  
from prompt_toolkit import PromptSession 
from prompt_toolkit.key_binding import KeyBindings 

class BaxBot:
    def __init__(self, memory: PersonalAgentMemory):        
        # 1. Load Configuration
        self.config = self._load_config("agent_assets/config.json")
        
        # 2. Load Prompt Template
        self.system_prompt_template = self._load_text("agent_assets/system_prompt.txt")

        self.summary_prompt_template = self._load_text("agent_assets/summary_system_prompt.txt")

        self.console = Console()
        self.memory = memory
        self.main_model = self.config["main_model"]
        self.summary_model = self.config["summary_model"]
        self.history_threshold = self.config["history_threshold"]
        self.baxbotTag = self.config["baxbot_tag"]

        self.console.print(f"")
        self.console.print(f"[bold blue]--- Initializing BaxBot ---[/bold blue]")
        self.console.print(f"\n{self.baxbotTag} Connecting to [bold cyan]{self.main_model}[/bold cyan] (for reasoning) and [bold cyan]{self.summary_model}[/bold cyan] (for summarization)...")

        self.session_history = []

        self.console.print(f"{self.baxbotTag} Registering [bold magenta]tools[/bold magenta]...")
        self.tool_registry = {
            "photo_sorter": PhotoSorter(),
            "system_stats": SystemStatsTool(),
            "memory_manager": MemoryManagerTool(),
        }
        self.console.print("")
        self.console.print(f"[bold green]--- BaxBot Ready ---[/bold green]")
        self.console.print(f"\n{self.baxbotTag} Hi! I'm BaxBot. What's on your mind?")

    def _load_config(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def _load_text(self, path):
        with open(path, 'r') as f:
            return f.read()

    def _print_markdown(self, text: str):
        self.console.print(f"\n{self.baxbotTag}")
        md = Markdown(text)
        self.console.print(md)

    def _archive_memory(self):
        self.console.print(f"\n{self.baxbotTag} Let me save this conversation to [bold]Long-Term Memory[/bold]...")

        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in self.session_history])
        
        summary_prompt = self.summary_prompt_template.replace("{{HISTORY_TEXT}}", history_text)
        
        summary_data = ollama.generate(model=self.summary_model, prompt=summary_prompt)
        summary = summary_data['response'].strip()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.memory.add_memory("info", f"Chat history {timestamp}", summary)
        
        self.session_history = []
        self.console.print(f"[bold green]{self.baxbotTag} Successfully saved the conversation![/bold green]")

    def run_tool(self, tool_name: str, args: dict = None):
        """Looks up the tool in the registry and executes it with provided args dictionary."""
        tool = self.tool_registry.get(tool_name)
        if tool:
            # args is now a dictionary (e.g., {"category": "info", "key": "bio", ...})
            return tool.execute(args) 
        return f"Error: Tool '{tool_name}' not found in registry."

    def chat(self, user_input: str):
        # 1. Retrieval Phase
        info_raw = self.memory.query_memory(user_input, "info")
        faith_raw = self.memory.query_memory(user_input, "faith")
        tone_raw = self.memory.query_memory(user_input, "tone")

        # We join list items with a newline so they appear as distinct points in the prompt
        info = "\n".join(info_raw) if isinstance(info_raw, list) else info_raw
        faith = "\n".join(faith_raw) if isinstance(faith_raw, list) else faith_raw
        tone = "\n".join(tone_raw) if isinstance(tone_raw, list) else tone_raw

        # 2. Construct Short-Term Memory String (STM)
        stm_context = ""
        if self.session_history:
            stm_context = "\n--- Recent Conversation ---\n" + "\n".join(
                [f"{m['role']}: {m['content']}" for m in self.session_history]
            )

        # 3. Reasoning Phase (Inject variables into template)
        # We use .replace() so we don't break the JSON curly braces in the text
        system_prompt = (
            self.system_prompt_template
            .replace("{{INFO}}", info)
            .replace("{{FAITH}}", faith)
            .replace("{{TONE}}", tone)
            .replace("{{STM_CONTEXT}}", stm_context)
        )

        full_prompt = f"{system_prompt}\n\nUser: {user_input}\nAssistant:"
        self.console.print(f"\n{self.baxbotTag} [italic]Thinking...[/italic]")
        
        response_data = ollama.generate(model=self.main_model, prompt=full_prompt)
        response = response_data['response'].strip()

        # 4. Action Phase (JSON Parsing)
        if "[TOOL_CALL]" in response:
            try:
                # Extract content between [TOOL_CALL] and [/TOOL_CALL]
                start_tag = "[TOOL_CALL]"
                end_tag = "[/TOOL_CALL]"
                
                start_idx = response.find(start_tag) + len(start_tag)
                end_idx = response.find(end_tag)
                
                json_str = response[start_idx:end_idx].strip()
                tool_data = json.loads(json_str)

                # self.console.print(f"tool_data: {tool_data}")  # Debugging line to inspect parsed JSON
                
                tool_name = tool_data["tool"]
                tool_args = tool_data.get("args", {}) # Default to empty dict if no args provided

                tool_result = self.run_tool(tool_name, tool_args)
                self.console.print(f"[bold magenta][{tool_name}]:[/bold magenta] {tool_result}")

                self.console.print(f"\n{self.baxbotTag}[italic]Thinking...[/italic]\n")

                # 5. Synthesis Phase
                synthesis_prompt = f"{full_prompt}\nTool Output: {tool_result}\nAssistant:"
                synthesis_data = ollama.generate(model=self.summary_model, prompt=synthesis_prompt)
                final_response = synthesis_data['response'].strip()
                
                self._print_markdown(final_response)
                
                # Record history
                self.session_history.append({"role": "user", "content": user_input})
                self.session_history.append({"role": "assistant", "content": final_response})

            except Exception as e:
                self.console.print(f"[bold red]Error parsing tool call: {e}[/bold red]")
                # Fallback if parsing fails
                self.session_history.append({"role": "user", "content": user_input})
                self.session_history.append({"role": "assistant", "content": response})
        else:
            # Standard Chat Response
            self._print_markdown(response)
            self.session_history.append({"role": "user", "content": user_input})
            self.session_history.append({"role": "assistant", "content": response})
        
        if len(self.session_history) >= self.history_threshold:
            self._archive_memory()


if __name__ == "__main__":
    main_console = Console()
    user_memory = PersonalAgentMemory()

    bot = BaxBot(user_memory)

    session = PromptSession()

    while True:
        try:
            main_console.print(f"\n[italic](Press 'Esc + Return' to submit and 'bye' to exit)[/italic]")
            query = session.prompt("Ask BaxBot: ", multiline=True)
        except EOFError:
            break

        if not query.strip():
            continue

        if query.lower() in ['exit', 'quit', 'bye', 'see ya']:
            if bot.session_history:
                bot._archive_memory()
            main_console.print(f"\n{bot.baxbotTag} Bye for now!\n")
            break

        bot.chat(query)
