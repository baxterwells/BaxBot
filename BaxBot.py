import os
import sys
import ollama
from memory_manager import PersonalAgentMemory 
from tools_library import PhotoSorter, SystemStatsTool, MemoryManagerTool
from rich.console import Console
from rich.markdown import Markdown  # <--- NEW: Import the Markdown parser
from prompt_toolkit import PromptSession # <--- NEW import for multi-line input
from prompt_toolkit.key_binding import KeyBindings # <--- NEW


main_model = "gemma4:26b-mlx"  # Ollama model for reasoning and tool orchestration
# main_model = "gemma2:27b"  # Ollama model for reasoning and tool orchestration
summary_model = "mistral"  # Ollama model for summarization tasks

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
        self.console.print(f"[bold green]{baxbotTag} Successfully saved the conversation![/bold green]")

    def run_tool(self, tool_name: str, args: list = None):
        """Looks up the tool in the registry and executes it with provided args."""
        tool = self.tool_registry.get(tool_name)
        if tool:
            return tool.execute(args) 
        return f"Error: Tool '{tool_name}' not found in registry."

    def chat(self, user_input: str):
        # 1. Retrieval Phase
        info = self.memory.query_memory(user_input, "info")
        faith = self.memory.query_memory(user_input, "faith")
        tone = self.memory.query_memory(user_input, "tone")

        # --- DEBUG LINE ---
        # print(f"\n[DEBUG] Retrieved Context:\n{info}\n...\n{faith}\n...\n{tone}\n...\n") 
        # ------------------

        # 2. Construct Short-Term Memory String (STM)
        stm_context = ""
        if self.session_history:
            stm_context = "\n--- Recent Conversation ---\n" + "\n".join(
                [f"{m['role']}: {m['content']}" for m in self.session_history]
            )

        # 3. Reasoning Phase
        system_prompt = f"""
        SUMMARY:
        - You are BaxBot, a personal AI companion. You have access to a set of tools and a long-term memory database.
        - You are an expert theologian and a skilled conversationalist. Your goal is to assist the user with their questions, tasks, and personal needs, exploring their faith when applicable.
        - Keep responses concise, unless the user requests more detail. Avoid unnecessary verbosity.
        - Don't atuomatically make the conversation about faith unless the user brings it up.
        - Feel free to use Markdown formatting in your responses, including headings, lists, and code blocks.
        
        PERSONAL INFO: {info}
        - This is information retrieved from your long-term memory. Use it to inform your responses.

        FAITH: {faith}
        - This is information about the user's faith, including written notes, references to the Bible, and other faith-based content.
        - If applicable and (the user is asking a faith-based question or is talking about their faith), use it to inform your responses. Otherwise, ignore it.

        TONE: {tone}
        - This tone (tone of voice) you should mimic and respond to the user in. The tone may change over time, so check the latest tone in memory.

        TOOLS:
        - To use a tool, you MUST respond with the exact syntax: CALL_TOOL: tool_name | arg1 | arg2
            - Available tools: photo_sorter, system_stats, memory_manager

        ABOUT THE TOOLS:
        1. photo_sorter: Scans a folder of images, detects objects, and sorts them into subfolders based on detected objects.
        2. system_stats: Returns current system statistics (CPU, memory, etc.) for the machine BaxBot is running on.
        3. memory_manager: Manages the personal memory vault. Can ingest new information or update existing entries.

        RULES FOR TOOLS:
        - Use the following context to inform whether to call a tool:
            - If the user asks about sorting photos or pictures, call 'photo_sorter' with no arguments.
            - If the user asks about system performance or stats, call 'system_stats' with no arguments.
            - If the user asks about storing a new memory, call 'memory_manager' with the following arguments: category, key, content.
                - If the user asks you to refresh or update your memory or to ingest memories, call 'memory_manager' with no arguments.
        - Otherwise, respond to the user naturally using the provided context and your unique tone.

        EXAMPLES OF TOOL CALLS:
        - CALL_TOOL: photo_sorter
        - CALL_TOOL: system_stats
        - CALL_TOOL: memory_manager | info | bio | I love dark mode
        - CALL_TOOL: memory_manager

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
    user_memory = PersonalAgentMemory()

    bot = BaxBot(main_model, user_memory)

    # Initialize the session
    session = PromptSession()

    while True:
        try:
            # multiline=True allows Enter to create new lines
            # The standard way to submit in this mode is Alt+Enter
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
