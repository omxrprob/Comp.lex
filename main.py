import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

try:
    # Attempt to import Google GenAI SDK and Rich library
    from google import genai
    from google.genai.errors import APIError
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt
except ImportError:
    print("Error: Required libraries not found.")
    print("Please install them using: pip install google-genai rich")
    sys.exit(1)

# --- Configuration and Setup ---
APP_NAME = "Comp.lex"
CONFIG_PATH = Path.home() / ".complex_config.json"
DEFAULT_MODEL = "gemini-2.5-flash"

# Initialize rich console for beautiful output
console = Console()

def load_config() -> Dict[str, Any]:
    """Loads configuration and history from the config file."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            console.print(f"[bold yellow]Warning:[/bold yellow] Configuration file corrupted. Starting fresh.")
            return {}
    return {}

def save_config(config: Dict[str, Any]):
    """Saves configuration and history to the config file."""
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
    except IOError as e:
        console.print(f"[bold red]Error:[/bold red] Could not save configuration: {e}")

def get_api_key(config: Dict[str, Any]) -> str:
    """Prompts the user for an API key if it's not in the config."""
    api_key = config.get("api_key")
    
    # Check if API key is present but invalid (e.g., placeholder)
    if not api_key or not api_key.startswith("AIza"):
        console.print(Panel(
            "[bold white]Welcome to Comp.lex[/bold white] (Complex AI for the Command Line)\n\n"
            "This application uses the Gemini API. Your API key will be saved locally in "
            f"[bold cyan]{CONFIG_PATH}[/bold cyan] and is [bold red]never[/bold red] transmitted to anyone but Google.\n\n"
            "[bold green]Instructions to get your key:[/bold green]\n"
            "1. Visit the Google AI Studio to get an API key."
            , title=f"[bold green]{APP_NAME} Setup[/bold green]", border_style="cyan"
        ))
        
        # Keep prompting until a key is provided
        while True:
            api_key = Prompt.ask("[bold yellow]Please enter your Gemini API Key[/bold yellow]").strip()
            if api_key.startswith("AIza"):
                config['api_key'] = api_key
                save_config(config)
                console.print("[bold green]API Key accepted and saved.[/bold green]")
                break
            else:
                console.print("[bold red]Invalid Key Format.[/bold red] Please ensure the key starts with 'AIza'.")

    return config['api_key']

class ComplexChat:
    """Manages the chat session, history, model, and settings."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("api_key")
        self.model_name = config.get("model", DEFAULT_MODEL)
        self.temperature = config.get("temperature", 0.7)
        self.system_instruction = config.get("system_instruction", "You are a concise, helpful assistant named Comp.lex, specialized in technical advice, coding, and developer tasks. Format all code responses in markdown code blocks.")
        self.search_grounding = config.get("search_grounding", False)
        self.history: List[Dict[str, str]] = config.get("history", [])

        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None # Will be set after key input

        self.chat_session = self._create_new_chat()

    def _create_new_chat(self):
        """Initializes a new chat session with current settings."""
        if not self.client:
            return None

        # The history needs to be reformatted from simple dict to genai.types.Content
        contents = [
            genai.types.Content(
                role=message["role"],
                parts=[genai.types.Part.from_text(message["text"])]
            )
            for message in self.history
        ]

        # Use gemini-2.5-flash-lite for tools and grounding
        model_to_use = "gemini-2.5-flash-lite" if self.search_grounding else self.model_name
        
        chat = self.client.chats.create(
            model=model_to_use,
            config={
                "system_instruction": self.system_instruction,
                "temperature": self.temperature,
                "tools": [{"google_search": {}}] if self.search_grounding else [],
            },
            history=contents
        )
        return chat

    def display_settings(self):
        """Displays the current model settings."""
        console.print(Panel(
            f"[bold yellow]Model:[/bold yellow] {self.model_name}\n"
            f"[bold yellow]Temperature:[/bold yellow] {self.temperature:.1f}\n"
            f"[bold yellow]Persona (System Instruction):[/bold yellow] {self.system_instruction}\n"
            f"[bold yellow]Google Search Grounding:[/bold yellow] {'[bold green]ON[/bold green]' if self.search_grounding else '[bold red]OFF[/bold red]'}\n"
            f"[bold yellow]History Length:[/bold yellow] {len(self.history)} messages"
            , title="[bold cyan]Current Settings[/bold cyan]", border_style="yellow"
        ))

    def process_command(self, user_input: str) -> bool:
        """Handles slash commands and returns True if a command was executed."""
        parts = user_input.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command == "/exit" or command == "/quit":
            console.print(f"\n[bold green]Exiting {APP_NAME}. Goodbye![/bold green]")
            return True
        
        elif command == "/help":
            console.print(Panel(
                "[bold cyan]Available Commands:[/bold cyan]\n"
                "[bold green]/model <name> [/bold green]- Switch model (e.g., flash, pro, gemini-2.5-flash).\n"
                "[bold green]/temp <0.0-1.0> [/bold green]- Set response creativity (e.g., /temp 0.9).\n"
                "[bold green]/persona <prompt> [/bold green]- Set the AI's role/system instruction (e.g., /persona Act as a Linux expert).\n"
                "[bold green]/search on/off [/bold green]- Toggle Google Search grounding for up-to-date info.\n"
                "[bold green]/history clear [/bold green]- Clear current chat history.\n"
                "[bold green]/settings [/bold green]- Display current settings.\n"
                "[bold green]/exit [/bold green]- Quit the application and save history."
                , title="[bold magenta]Help Menu[/bold magenta]", border_style="magenta"
            ))

        elif command == "/settings":
            self.display_settings()

        elif command == "/model":
            if not arg:
                console.print("[bold red]Error:[/bold red] Please specify a model name (e.g., /model pro).")
                return False
            
            # Simple mapping for user convenience
            model_map = {
                "flash": "gemini-2.5-flash", 
                "pro": "gemini-2.5-pro",
                "lite": "gemini-2.5-flash-lite",
            }
            new_model = model_map.get(arg.lower(), arg)
            
            # Basic validation
            if not new_model.startswith("gemini"):
                console.print(f"[bold red]Error:[/bold red] '{arg}' is not a recognized model name. Use gemini-2.5-flash, gemini-2.5-pro, etc.")
                return False

            self.model_name = new_model
            self.config["model"] = self.model_name
            self.chat_session = self._create_new_chat()
            console.print(f"[bold green]Model changed to:[/bold green] {self.model_name}. Chat restarted.")

        elif command == "/temp":
            try:
                temp = float(arg)
                if 0.0 <= temp <= 1.0:
                    self.temperature = temp
                    self.config["temperature"] = self.temperature
                    self.chat_session = self._create_new_chat()
                    console.print(f"[bold green]Temperature set to:[/bold green] {self.temperature}. Chat restarted.")
                else:
                    console.print("[bold red]Error:[/bold red] Temperature must be between 0.0 and 1.0.")
            except ValueError:
                console.print("[bold red]Error:[/bold red] Invalid temperature value.")

        elif command == "/persona":
            if not arg:
                console.print("[bold red]Error:[/bold red] Please provide a persona prompt.")
                return False
            self.system_instruction = arg
            self.config["system_instruction"] = self.system_instruction
            self.chat_session = self._create_new_chat()
            console.print(f"[bold green]Persona updated.[/bold green] Chat restarted with new role: '{arg}'")

        elif command == "/search":
            if arg.lower() in ["on", "true"]:
                self.search_grounding = True
                console.print("[bold green]Google Search Grounding ENABLED.[/bold green] Responses will be grounded in current web data. Note: The model will be temporarily set to 'gemini-2.5-flash-lite' when search is enabled to optimize for tool use.")
            elif arg.lower() in ["off", "false"]:
                self.search_grounding = False
                console.print("[bold yellow]Google Search Grounding DISABLED.[/bold yellow] Using primary model for general knowledge.")
            else:
                console.print("[bold red]Error:[/bold red] Use '/search on' or '/search off'.")
            
            self.config["search_grounding"] = self.search_grounding
            self.chat_session = self._create_new_chat() # Recreate chat to apply grounding tool

        elif command == "/history":
            if arg.lower() == "clear":
                self.history = []
                self.config["history"] = []
                self.chat_session = self._create_new_chat()
                console.print("[bold yellow]Chat history cleared.[/bold yellow] Chat session restarted.")
            else:
                console.print("[bold red]Error:[/bold red] Use '/history clear' to clear the conversation history.")
        
        else:
            console.print(f"[bold red]Unknown command:[/bold red] {command}. Use /help for a list of commands.")

        return False

    def start_chat_loop(self):
        """The main interactive chat loop."""
        
        # Display the prompt and a welcome message with settings
        self.display_settings()
        
        while True:
            try:
                user_input = Prompt.ask(f"\n[bold cyan]{os.getlogin()}@[/bold cyan][bold magenta]{self.model_name.split('-')[-1].upper()}[/bold magenta]").strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith("/"):
                    if self.process_command(user_input):
                        break
                    continue
                
                # Add user message to history before sending
                self.history.append({"role": "user", "text": user_input})

                # Stream the response
                console.print("\n[bold green]Comp.lex[/bold green]: ", end="")
                full_response = ""
                
                # Check which model to use based on grounding setting
                model_to_use = "gemini-2.5-flash-lite" if self.search_grounding else self.model_name

                # Start streaming process
                response_stream = self.client.models.generate_content_stream(
                    model=model_to_use,
                    contents=[
                        genai.types.Content(
                            role="user",
                            parts=[genai.types.Part.from_text(user_input)]
                        )
                    ],
                    config={
                        "system_instruction": self.system_instruction,
                        "temperature": self.temperature,
                        "tools": [{"google_search": {}}] if self.search_grounding else [],
                    }
                )
                
                for chunk in response_stream:
                    if chunk.text:
                        console.print(chunk.text, end="")
                        sys.stdout.flush()
                        full_response += chunk.text
                
                # Print newline after streaming is complete
                console.print()

                # Save the model's full response and update history
                if full_response:
                    self.history.append({"role": "model", "text": full_response})
                    self.config["history"] = self.history
                    save_config(self.config)

            except APIError as e:
                console.print(Panel(
                    f"[bold red]API Error:[/bold red] A problem occurred while communicating with the Gemini API.\n"
                    f"Check your API key and network connection. Error details: [yellow]{e}[/yellow]"
                    , title="[bold red]Gemini Error[/bold red]", border_style="red"
                ))
            except KeyboardInterrupt:
                console.print("\n[bold green]Exiting Comp.lex.[/bold green]")
                break
            except Exception as e:
                console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")
                time.sleep(1)


def main():
    """Main function to initialize and run the chat application."""
    console.print(f"[bold blue]Initializing {APP_NAME} CLI...[/bold blue]")
    
    config = load_config()
    
    # 1. Get and save the API Key
    try:
        get_api_key(config)
    except Exception as e:
        console.print(f"[bold red]Fatal Error during setup:[/bold red] {e}")
        sys.exit(1)

    # 2. Initialize and start the chat
    chat_app = ComplexChat(config)
    if chat_app.client:
        chat_app.start_chat_loop()
    
    # 3. Final save
    save_config(chat_app.config)

if __name__ == "__main__":
    main()

