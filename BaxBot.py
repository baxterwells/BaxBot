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
    """The central logic that coordinates Memory, UI, Tools, and LLM using ReAct architecture."""
    def __init__(self, config: dict, memory, ui, tools):
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

    def _parse_tool_call(self, response: str) -> dict:
        """Extracts the tool name and arguments from the [TOOL_CALL] tags."""
        start_tag = "[TOOL_CALL]"
        end_tag = "[/TOOL_CALL]"
        
        start_idx = response.find(start_tag)
        end_idx = response.find(end_tag)

        if start_idx == -1 or end_idx == -1:
            raise ValueError("Missing tool tags in response.")

        content_start = start_idx + len(start_tag)
        json_str = response[content_start:end_idx].strip()

        if not json_str:
            raise ValueError("The tool call block is empty.")

        tool_data = json.loads(json_str)
        
        if not isinstance(tool_data, dict):
            raise ValueError("Expected a JSON object for the tool call.")

        name = tool_data.get("tool")
        args = tool_data.get("args", {})

        if not name:
            raise ValueError("JSON missing the required 'tool' key.")

        return {"name": name, "args": args}

    def chat(self, user_input: str):
        """The main ReAct Loop: Thought -> Action -> Observation."""
        # 1. Record the user input in history
        self.session_history.append({"role": "user", "content": user_input})

        # Loop control variables
        max_steps = 5  # Prev-ents infinite loops if the agent gets stuck
        steps = 0
        loop_complete = False

        while steps < max_steps and not loop_complete:
            # 2. Context Construction (Build history string for the current loop iteration)
            # We include everything in the history so the agent sees its own previous thoughts/actions
            stm_context = "\n--- Recent Conversation ---\n" + "\n".join(
                [f"{m['role']}: {m['content']}" for m in self.session_history]
            )

            # 3. Retrieve Long-term Memory (Info/Faith/Tone)
            retrieved_items = self.memory.query_all_memory(user_input)
            info, faith, tone = [], [], []
            for item in retrieved_items:
                if item['category'] == "info": info.append(item['content'])
                elif item['category'] == "faith": faith.append(item['content'])
                elif item['category'] == "tone": tone.append(item['content'])

            # 4. Assemble the current System Prompt
            system_prompt = self._get_system_prompt(info, faith, tone, stm_context)
            full_prompt = f"{system_prompt}\n\n{stm_context}\nAssistant:"

            # 5. LLM Generation
            self.ui.print_status(f"BaxBot is thinking (Step {steps+1}/{max_steps})...")
            response_data = ollama.generate(model=self.config["main_model"], prompt=full_prompt)
            response = response_data['response'].strip()

            # 6. Decision Logic: Is this a Tool Call or a Final Answer?
            if "[TOOL_CALL]" in response:
                try:
                    # Step A: Parse the tool call
                    tool_info = self._parse_tool_call(response)
                    name = tool_info["name"]
                    args = tool_info["args"]

                    # Step B: Execute the tool
                    self.ui.print_status(f"Running [{name}]...")
                    result = self.tools.execute(name, args)
                    self.ui.print_tool_output(name, result)

                    # Step C: Record the action and the observation back into history
                    # We use 'system' role for the observation so the agent treats it as fact
                    self.session_history.append({"role": "assistant", "content": response})
                    self.session_history.append({"role": "system", "content": f"OBSERVATION: {result}"})
                    
                    steps += 1 # Continue the loop to let the agent process the observation
                except Exception as e:
                    self.ui.print_error(f"Tool Execution Error: {e}")
                    self.session_history.append({"role": "assistant", "content": f"Error: {str(e)}"})
                    break # Exit loop on error to prevent infinite loops
            else:
                # This is a final response (no [TOOL_CALL] found)
                self.ui.print_bot_message(response)
                self.session_history.append({"role": "assistant", "content": response})
                loop_complete = True

        # 7. Post-Chat: Check for archiving
        if len(self.session_history) >= self.history_threshold:
            self._archive_memory()

    def _archive_memory(self):
        """Saves summarized history to long-term memory."""
        self.ui.print_status("Archiving conversation to memory...")
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in self.session_history])
        
        summary_prompt = f"Summarize this conversation for long-term storage:\n{history_text}"
        summary_data = ollama.generate(model=self.config["summary_model"], prompt=summary_prompt)
        summary = summary_data['response'].strip()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M")
        self.memory.add_memory("info", f"Chat history {timestamp}", summary)
        self.session_history = []
        self.ui.print_status("Archive complete.")



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
                ui.print_bot_message("Bye for now!\n\n")
                break
            if not query.strip(): continue
            
            bot.chat(query)
        except EOFError:
            break
