import ollama
import sys
from rich.console import Console
from rich.markdown import Markdown  # Added for Markdown rendering

def run_session(model_name):
    console = Console()
    
    console.print(f"\n[bold gold1]D&D Expert Mode Active[/bold gold1]")
    console.print(f"[italic]Using Model: [bold cyan]{model_name}[/bold cyan][/italic]\n")
    console.print("[italic](Type 'exit' to return to BaxBot)[/italic]")

    system_prompt = (
        "You are a highly advanced D&D 5th Edition Rules Expert and Dungeon Master. "
        "Your goal is to provide mathematically and mechanically accurate advice. "
        "STRICT RULES:\n"
        "1. Always verify spell levels. If a user mentions a character level, ensure suggested spells are legal.\n"
        "2. Use official terminology (e.g., 'Saving Throw', 'Armor Class', 'Ability Check', 'Cantrip').\n"
        "3. If a user asks a question unrelated to D&D, politely redirect them back to D&D rules.\n"
        "4. Be extremely precise with mechanics, including duration, range, and components.\n"
        "5. If a rule is controversial or has multiple interpretations, mention the most common one."
    )

    chat_history = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            user_input = input("\nAsk D&D Expert: ")

            if user_input.lower().strip() in ["exit", "quit", "bye"]:
                console.print("\nExiting [bold gold1]D&D Expert Mode[/bold gold1]...")
                break

            if not user_input.strip():
                continue

            chat_history.append({"role": "user", "content": user_input})

            # LLM Call
            response = ollama.chat(model=model_name, messages=chat_history)
            bot_message = response['message']['content']
            
            # --- UPDATED MARKDOWN LOGIC ---
            # 1. Print the Bot's Name/Tag
            console.print(f"\n[bold gold1]D&D Expert:[/bold gold1]")
            
            # 2. Convert raw text to Markdown and print it
            md = Markdown(bot_message)
            console.print(md)
            # -------------------------------

            chat_history.append({"role": "assistant", "content": bot_message})

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
            break

if __name__ == "__main__":
    # We take the model name from the command line argument
    # sys.argv[1] is the first argument passed to the script
    target_model = sys.argv[1] if len(sys.argv) > 1 else "gemma2:27b"
    run_session(target_model)
