import os
import json
import datetime
import ollama
from rich.console import Console
from rich.markdown import Markdown

# --- 1. THE VOICE/EYES (UI HANDLER) ---
class BaxUI:
    """Handles all interactions with the terminal/console."""
    def __init__(self, console: Console, config: dict):
        self.console = console
        self.tag = config["baxbot_tag"]

    def print_header(self, text: str):
        self.console.print(f"\n[bold green]---{text}---[/bold green]")

    def print_bot_message(self, text: str):
        self.console.print(f"\n{self.tag}")
        md = Markdown(text)
        self.console.print(md)

    def print_status(self, message: str):
        self.console.print(f"\n{self.tag} \n[italic]{message}[/italic]")

    def print_tool_output(self, tool_name: str, result: str):
        self.console.print(f"[bold magenta][{tool_name}]:[/bold magenta] {result}")

    def print_error(self, message: str):
        self.console.print(f"[bold red]Error: {message}[/bold red]")


# --- 2. THE HANDS (TOOL REGISTRY) ---
class BaxTools:
    """Handles the execution of all available tools."""
    def __init__(self, tool_instances: dict):
        self.registry = tool_instances

    def execute(self, tool_name: str, args: dict) -> str:
        tool = self.registry.get(tool_name)
        if tool:
            return tool.execute(args)
        return f"Error: Tool '{tool_name}' not found."


# --- 3. THE BRAIN (ORCHESTRATOR) ---
class BaxOrchestrator:
    """The central logic that coordinates Memory, UI, Tools, and LLM."""
    def __init__(self, config: dict, memory, ui: BaxUI, tools: BaxTools):
        self.config = config
        self.memory = memory
        self.ui = ui
        self.tools = tools
        
        # Load the long prompt template once during init
        with open(config["prompt_path"], 'r') as f:
            self.prompt_template = f.read()

        self.session_history = []
        self.history_threshold = config["history_threshold"]

    def _get_system_prompt(self, info_raw, faith_raw, tone_raw, stm_context) -> str:
        """Assembles the prompt by injecting context into the template."""
        info = "\n".join(info_raw) if isinstance(info_raw, list) else info_raw
        faith = "\n".join(faith_raw) if isinstance(faith_raw, list) else faith_raw
        tone = "\n".join(tone_raw) if isinstance(tone_raw, list) else tone_raw
        return (
            self.prompt_template
            .replace("{{INFO}}", info)
            .replace("{{FAITH}}", faith)
            .replace("{{TONE}}", tone)
            .replace("{{STM_CONTEXT}}", stm_context)
        )

    def _archive_memory(self):
        self.ui.print_status("Let me save this conversation to Long-Term Memory...")
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in self.session_history])
        
        summary_prompt = f"Summarize this:\n{history_text}" # Simplified for skeleton
        summary_data = ollama.generate(model=self.config["summary_model"], prompt=summary_prompt)
        summary = summary_data['response'].strip()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.memory.add_memory("info", f"Chat history {timestamp}", summary)
        self.session_history = []
        self.ui.print_status("Successfully saved!")

    def chat(self, user_input: str):
        # 1. SINGLE Unified Retrieval Phase
        # We get a list of dicts: [{'content': '...', 'category': 'info'}, ...]
        retrieved_items = self.memory.query_all_memory(user_input)

        # 2. Local Categorization (Sorting the 'Enriched' items into their buckets)
        info = []
        faith = []
        tone = []

        for item in retrieved_items:
            if item['category'] == "info":
                info.append(item['content'])
            elif item['category'] == "faith":
                faith.append(item['content'])
            elif item['category'] == "tone":
                tone.append(item['content'])

        # 3. Context Construction (The rest of your code remains exactly the same)
        stm_context = ""
        if self.session_history:
            stm_context = "\n--- Recent Conversation ---\n" + "\n".join(
                [f"{m['role']}: {m['content']}" for m in self.session_history]
            )

        # 4. Prompt Assembly (Pass the lists we just built)
        system_prompt = self._get_system_prompt(info, faith, tone, stm_context)
        full_prompt = f"{system_prompt}\n\nUser: {user_input}\nAssistant:"

        self.ui.print_status("Thinking...")

        
        # 4. LLM Call
        response_data = ollama.generate(model=self.config["main_model"], prompt=full_prompt)
        response = response_data['response'].strip()

        # 5. Action/Synthesis Logic
        if "[TOOL_CALL]" in response:
            self._handle_tool_call(full_prompt, response, user_input)
        else:
            self._handle_standard_chat(response, user_input)

        # 6. Check for archiving
        if len(self.session_history) >= self.history_threshold:
            self._archive_memory()

    def _handle_tool_call(self, full_prompt, response, user_input):
        try:
            # 1. DEFENSIVE TAG SEARCH
            start_tag = "[TOOL_CALL]"
            end_tag = "[/TOOL_CALL]"
            
            start_idx = response.find(start_tag)
            end_idx = response.find(end_tag)

            if start_idx == -1 or end_idx == -1:
                raise ValueError(f"Missing tool tags. Found Start: {start_idx}, End: {end_idx}")

            # 2. EXTRACT STRING
            # Move index to the end of the start tag
            content_start = start_idx + len(start_tag)
            json_str = response[content_start:end_idx].strip()

            if not json_str:
                raise ValueError("The tool call block is empty.")

            # 3. SAFE JSON PARSING
            try:
                tool_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format: {e}")

            # 4. STRUCTURE & KEY VALIDATION
            # Check if it's a dictionary (LLMs sometimes accidentally return lists)
            if not isinstance(tool_data, dict):
                raise ValueError(f"Expected a JSON object (dict), but got {type(tool_data).__name__}")

            # Use .get() to prevent KeyError: 0
            name = tool_data.get("tool")
            args = tool_data.get("args", {})

            if not name:
                raise ValueError(f"JSON missing the required 'tool' key. Found: {list(tool_data.keys())}")

            # 5. EXECUTION
            self.ui.print_status(f"Executing {name}...")
            result = self.tools.execute(name, args)
            self.ui.print_tool_output(name, result)

            # 6. SYNTHESIS
            synthesis_prompt = f"{full_prompt}\nTool Output: {result}\nAssistant:"
            s_data = ollama.generate(model=self.config["summary_model"], prompt=synthesis_prompt)
            final_resp = s_data['response'].strip()

            self.ui.print_bot_message(final_resp)
            self.session_history.append({"role": "user", "content": user_input})
            self.session_history.append({"role": "assistant", "content": final_resp})

        except Exception as e:
            # This catches our custom ValueErrors and any unexpected crashes
            self.ui.print_error(f"Parsing failed: {e}")
            self.ui.print_status("Check your prompt format.")
            
            # Fallback so the conversation doesn't die
            self.session_history.append({"role": "user", "content": user_input})
            self.session_history.append({"role": "assistant", "content": response})

    def _handle_standard_chat(self, response, user_input):
        self.ui.print_bot_message(response)
        self.session_history.append({"role": "user", "content": user_input})
        self.session_history.append({"role": "assistant", "content": response})


# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    from memory_manager import PersonalAgentMemory # Assume import exists
    from tools_library import PhotoSorter, SystemStatsTool, MemoryManagerTool, DnDExpertTool # Assume imports exist

    # 1. Load Configuration
    with open("agent_assets/config.json", 'r') as f:
        config = json.load(f)
        config["prompt_path"] = "agent_assets/system_prompt.txt"

    # 2. Initialize Dependencies
    console = Console()
    memory = PersonalAgentMemory()
    ui = BaxUI(console, config)
    
    tools_map = {
        "photo_sorter": PhotoSorter(),
        "system_stats": SystemStatsTool(),
        "memory_manager": MemoryManagerTool(),
        "dnd_expert": DnDExpertTool(),
    }
    tools = BaxTools(tools_map)

    # 3. Initialize Orchestrator
    bot = BaxOrchestrator(config, memory, ui, tools)

    # 4. Start Session
    ui.print_header("Initializing BaxBot")
    ui.print_status(f"Connecting to [bold cyan]{config['main_model']}[/bold cyan] for reasoning and [bold cyan]{config['summary_model']}[/bold cyan] for summarization...")
    ui.print_bot_message("Hi! I'm BaxBot. What's on your mind?")

    from prompt_toolkit import PromptSession
    session = PromptSession()

    while True:
        try:
            console.print("\n[italic](Press 'Esc + Return' to submit, 'bye' to exit)[/italic]")
            query = session.prompt("Ask BaxBot: ", multiline=True)
            if query.lower() in ['exit', 'quit', 'bye', 'see ya']:
                if bot.session_history: bot._archive_memory()
                ui.print_bot_message("Bye for now!\n")
                break
            if not query.strip(): continue
            
            bot.chat(query)
        except EOFError:
            break
