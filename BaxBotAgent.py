import os
import ollama
from memory_manager import PersonalAgentMemory 
from tools_library import FamilyPhotoSorter, SystemStatsTool, MemoryManagerTool, DnDCharacterSheetTool
from rich.console import Console
from rich.markdown import Markdown  # <--- NEW: Import the Markdown parser

main_model = "gemma4:26b-mlx"  # Ollama model for reasoning and tool orchestration
# main_model = "gemma2:27b"  # Ollama model for reasoning and tool orchestration
summary_model = "mistral"  # Ollama model for summarization tasks

baxbotTag = "[bold white][󱚤 BaxBot]:[/bold white]"

class BaxBot:
    def __init__(self, model_name: str, memory: PersonalAgentMemory):
        self.console = Console()
        self.console.print("")
        self._print_markdown_vanilla("### Initializing BaxBot...")
        self.console.print(f"\n{baxbotTag} Connecting to [bold cyan]{model_name}[/bold cyan] (for reasoning) and [bold cyan]{summary_model}[/bold cyan] (for summarization)...")
        
        self.model_name = model_name
        self.memory = memory

        self.session_history = [] 
        self.history_threshold = 10 

        self.console.print(f"{baxbotTag} Registering [bold magenta]tools[/bold magenta]...")
        self.tool_registry = {
            "family_photo_sorter": FamilyPhotoSorter(),
            "system_stats": SystemStatsTool(),
            "memory_manager": MemoryManagerTool(),
            "dnd_character_sheet": DnDCharacterSheetTool(),
        }
        self.console.print("")
        self._print_markdown_vanilla("# BaxBot ready")
        self.console.print(f"\n{baxbotTag} Hi! I'm BaxBot. What's on your mind?")

    def _print_markdown_vanilla(self, text: str):
            """Helper method to render text as beautiful Markdown."""
            # 1. Parse the text as Markdown and print it
            md = Markdown(text)
            self.console.print(md)

    def _print_markdown(self, text: str):
        """Helper method to render text as beautiful Markdown."""
        # 1. Print the identity prefix first
        self.console.print(f"\n{baxbotTag}")
        # 2. Parse the text as Markdown and print it
        md = Markdown(text)
        self.console.print(md)

    def _archive_memory(self):
        """Summarizes current session history and pushes it to Long-Term Memory (ChromaDB)."""
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

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.memory.add_memory("info", f"Chat history {timestamp}", summary)
        
        self.session_history = []
        self.console.print(f"\n[bold green]{baxbotTag} Successfully saved the conversation![/bold green]")

    def run_tool(self, tool_name: str, args: list = None):
        """Looks up the tool in the registry and executes it with provided args."""
        tool = self.tool_registry.get(tool_name)
        if tool:
            return tool.execute(args) 
        return f"Error: Tool '{tool_name}' not found in registry."

    def chat(self, user_input: str):
        # 1. Retrieval Phase
        info = self.memory.query_memory(user_input, "info")
        tone = self.memory.query_memory(user_input, "tone")
        context = "\n".join(info + tone)

        # --- DEBUG LINE ---
        # print(f"\n[DEBUG] Retrieved Context: {context}\n") 
        # ------------------

        # 2. Construct Short-Term Memory String (STM)
        stm_context = ""
        if self.session_history:
            stm_context = "\n--- Recent Conversation ---\n" + "\n".join(
                [f"{m['role']}: {m['content']}" for m in self.session_history]
            )

        # 3. Reasoning Phase
        system_prompt = f"""
        You are BaxBot, a personal AI assistant.
        
        PERSONAL CONTEXT:
        {context}

        RULES:
        - To use a tool, you MUST respond with the exact syntax: CALL_TOOL: tool_name | arg1 | arg2
        - Example: CALL_TOOL: memory_manager | info | bio | I love dark mode
        - Example: CALL_TOOL: dnd_character_sheet | update_hp | -5
        - Available tools: family_photo_sorter, system_stats, memory_manager, dnd_character_sheet
        - Use the following context to inform whether to call a tool:
            - If the user asks about sorting family photos, call 'family_photo_sorter'.
            - If the user asks about system performance or stats, call 'system_stats'.
            - If the user asks about memory management or vault updates, call 'memory_manager'.
            - If the user asks about, talks about, or references their DnD character, call 'dnd_character_sheet'.
                - When calling 'dnd_character_sheet', and the user asks for information about their character, use the 'get_sheet' command as the argument.
        - Otherwise, respond to the user naturally using the provided context and your unique tone.

        ABOUT THE TOOLS:
        1. family_photo_sorter: Scans a folder of images, detects objects, and sorts them into subfolders based on detected objects.
        2. system_stats: Returns current system statistics (CPU, memory, etc.) for the machine BaxBot is running on.
        3. memory_manager: Manages the personal memory vault. Can ingest new information or update existing entries.
        4. dnd_character_sheet: Manages the current active DnD character. Can retrieve the full sheet, update attributes, adjust HP, or add weapons.
        """

        full_prompt = f"{system_prompt}{stm_context}\n\nUser: {user_input}\nAssistant:"
        self.console.print(f"\n{baxbotTag} [italic]Thinking...[/italic]")
        
        response_data = ollama.generate(model=self.model_name, prompt=full_prompt)
        response = response_data['response'].strip()

        # 4. Action Phase (Dynamic Parsing)
        if "CALL_TOOL:" in response:
            raw_call = response.split("CALL_TOOL:")[1].strip()
            parts = [p.strip() for p in raw_call.split("|")]
            
            tool_name = parts[0]
            tool_args = parts[1:] 

            tool_result = self.run_tool(tool_name, tool_args)
            self.console.print(f"[bold magenta][{tool_name}]:[/bold magenta] {tool_result}")

            self.console.print(f"\n{baxbotTag}[italic]Thinking...[/italic]\n")

            # 5. Synthesis Phase
            synthesis_prompt = f"{full_prompt}\nTool Output: {tool_result}\nAssistant:"
            synthesis_data = ollama.generate(model=self.model_name, prompt=synthesis_prompt)
            final_response = synthesis_data['response'].strip()
            
            # USE THE NEW MARKDOWN HELPER HERE
            self._print_markdown(final_response)

            self.session_history.append({"role": "user", "content": user_input})
            self.session_history.append({"role": "assistant", "content": final_response})
        else:
            # USE THE NEW MARKDOWN HELPER HERE
            self._print_markdown(response)
            
            self.session_history.append({"role": "user", "content": user_input})
            self.session_history.append({"role": "assistant", "content": response})
        
        if len(self.session_history) >= self.history_threshold:
            self._archive_memory()


if __name__ == "__main__":
    main_console = Console()
    
    MY_MODEL_NAME = main_model
    user_memory = PersonalAgentMemory()
    bot = BaxBot(MY_MODEL_NAME, user_memory)

    while True:
        query = input("\nAsk BaxBot (or 'bye'): ")
        
        if query.lower() in ['exit', 'quit', 'bye', 'see ya']:
            if bot.session_history:
                bot._archive_memory()

            main_console.print(f"\n{baxbotTag} Bye for now!\n")
            break
        bot.chat(query)
