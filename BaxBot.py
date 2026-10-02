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

main_model = "gemma4:26b-mlx"  
summary_model = "mistral"  

baxbotTag = "[bold white][󱚤 BaxBot]:[/bold white]"

class BaxBot:
    def __init__(self, model_name: str, memory: PersonalAgentMemory):
        self.console = Console()
        self.console.print("")
        self._print_markdown_vanilla("# Initializing BaxBot")
        self.console.print(f"\n{baxbotTag} Connecting to [bold cyan]{model_name}[/bold cyan] (for reasoning) and [bold cyan]{summary_model}[/bold cyan] (for summarization)...")
        
        self.model_name = model_name
        self.memory = memory

        self.session_history = [] 
        self.history_threshold = 10 

        self.console.print(f"{baxbotTag} Registering [bold magenta]tools[/bold magenta]...")
        self.tool_registry = {
            "photo_sorter": PhotoSorter(),
            "system_stats": SystemStatsTool(),
            "memory_manager": MemoryManagerTool(),
        }
        self.console.print("")
        self._print_markdown_vanilla("# BaxBot ready")
        self.console.print(f"\n{baxbotTag} Hi! I'm BaxBot. What's on your mind?")

    def _print_markdown_vanilla(self, text: str):
            md = Markdown(text)
            self.console.print(md)

    def _print_markdown(self, text: str):
        self.console.print(f"\n{baxbotTag}")
        md = Markdown(text)
        self.console.print(md)

    def _archive_memory(self):
        self.console.print(f"\n{baxbotTag} Let me save this conversation to [bold]Long-Term Memory[/bold]...")

        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in self.session_history])
        
        summary_prompt = f"""
        You are a memory clerk. Summarize the following conversation into a concise, 
        fact-dense paragraph that captures important personal details, preferences, 
        and ongoing tasks. This will be used for future retrieval.
        
        CONVERSATION:
        {history_text}
        
        SUMMARY:
        """
        
        summary_data = ollama.generate(model=summary_model, prompt=summary_prompt)
        summary = summary_data['response'].strip()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.memory.add_memory("info", f"Chat history {timestamp}", summary)
        
        self.session_history = []
        self.console.print(f"[bold green]{baxbotTag} Successfully saved the conversation![/bold green]")

    def run_tool(self, tool_name: str, args: dict = None):
        """Looks up the tool in the registry and executes it with provided args dictionary."""
        tool = self.tool_registry.get(tool_name)
        if tool:
            # args is now a dictionary (e.g., {"category": "info", "key": "bio", ...})
            return tool.execute(args) 
        return f"Error: Tool '{tool_name}' not found in registry."

    def chat(self, user_input: str):
        # 1. Retrieval Phase
        info = self.memory.query_memory(user_input, "info")
        faith = self.memory.query_memory(user_input, "faith")
        tone = self.memory.query_memory(user_input, "tone")

        # 2. Construct Short-Term Memory String (STM)
        stm_context = ""
        if self.session_history:
            stm_context = "\n--- Recent Conversation ---\n" + "\n".join(
                [f"{m['role']}: {m['content']}" for m in self.session_history]
            )

        # 3. Reasoning Phase
        # UPDATED SYSTEM PROMPT FOR JSON TOOL CALLING
        system_prompt = f"""
        SUMMARY:
        - You are BaxBot, a personal AI companion. You have access to a set of tools and a long-term memory database.
        - You are an expert theologian and a skilled conversationalist. Your goal is to assist the user with their questions, tasks, and personal needs, exploring their faith when applicable.
        - Keep responses concise, unless the user requests more detail. Avoid unnecessary verbosity.
        - Don't automatically make the conversation about faith unless the user brings it up.
        - Feel free to use Markdown formatting in your responses, including headings, lists, and code blocks.
        
        PERSONAL INFO: {info}
        FAITH: {faith}
        TONE: {tone}

        TOOLS:
        - To use a tool, you MUST respond with a JSON block wrapped in [TOOL_CALL] tags.
        - Format: [TOOL_CALL] {{"tool": "tool_name", "args": {{"arg_name": "value"}}}} [/TOOL_CALL]

        AVAILABLE TOOLS:
        1. photo_sorter: Scans/sorts images. 
           - Usage: [TOOL_CALL] {{"tool": "photo_sorter", "args": {{}}}} [/TOOL_CALL]
        2. system_stats: Returns CPU/Memory stats.
           - Usage: [TOOL_CALL] {{"tool": "system_stats", "args": {{}}}} [/TOOL_CALL]
        3. memory_manager: Manages the personal memory vault.
           - Usage: [TOOL_CALL] {{"tool": "memory_manager", "args": {{"category": "info", "key": "bio", "content": "text"}}}} [/TOOL_CALL]
           - Usage (refresh): [TOOL_CALL] {{"tool": "memory_manager", "args": {{}}}} [/TOOL_CALL]

        EXAMPLES:
        - User: "Sort my pictures" -> [TOOL_CALL] {{"tool": "photo_sorter", "args": {{}}}} [/TOOL_CALL]
        - User: "Remember that I love dark mode" -> [TOOL_CALL] {{"tool": "memory_manager", "args": {{"category": "info", "key": "preference", "content": "I love dark mode"}}}} [/TOOL_CALL]
        """

        full_prompt = f"{system_prompt}{stm_context}\n\nUser: {user_input}\nAssistant:"
        self.console.print(f"\n{baxbotTag} [italic]Thinking...[/italic]")
        
        response_data = ollama.generate(model=self.model_name, prompt=full_prompt)
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
                
                tool_name = tool_data["tool"]
                tool_args = tool_data.get("args", {}) # Default to empty dict if no args provided

                tool_result = self.run_tool(tool_name, tool_args)
                self.console.print(f"[bold magenta][{tool_name}]:[/bold magenta] {tool_result}")

                self.console.print(f"\n{baxbotTag}[italic]Thinking...[/italic]\n")

                # 5. Synthesis Phase
                synthesis_prompt = f"{full_prompt}\nTool Output: {tool_result}\nAssistant:"
                synthesis_data = ollama.generate(model=self.model_name, prompt=synthesis_prompt)
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
    user_memory = PersonalAgent_Memory() # Assuming this is correct based on your import

    bot = BaxBot(main_model, user_memory)

    session = PromptSession()

    while True:
        try:
            query = session.prompt("\nAsk BaxBot (Press 'Esc+Return' to submit or 'bye'): ", multiline=True)
        except EOFError:
            break

        if not query.strip():
            continue

        if query.lower() in ['exit', 'quit', 'bye', 'see ya']:
            if bot.session_history:
                bot._archive_memory()
            main_console.print(f"\n{baxbotTag} Bye for now!\n")
            break

        bot.chat(query)
