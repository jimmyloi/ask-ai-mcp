# Add platform and subprocess imports
import platform
import subprocess
import uuid
# import subprocess # No longer needed - Re-add it!
import threading
# import json # Keep json for potential future use or other parts
# import shlex # No longer needed
from typing import List, Dict, Any, Optional

# Import LiteLLM
import litellm

# Import Context from fastmcp
from fastmcp import FastMCP, Context

# --- Configuration ---
# Assuming these are the "frontier" models
# NOTE: Verify these model strings are compatible with litellm.
# You might need formats like "openai/o1", "claude-3-opus-20240229", "gemini/gemini-1.5-flash", etc.
# Check litellm documentation for the correct identifiers.
DEFAULT_MODELS = [
    "openai/o1", # Example, might need changing
    "anthropic/claude-3-5-sonnet-20240620", # Example, might need changing
    "gemini/gemini-2.0-flash", # Example, might need changing
    "bedrock/us.deepseek.r1-v1:0" # Example, might need changing
]
# Map tool names to specific model identifiers from DEFAULT_MODELS
# We'll use the first one found for each provider as a default for the specific tools
MODEL_MAP = {
    "gpt": "openai/o1", # Default GPT model
    "claude": "anthropic/claude-3-5-sonnet-20240620", # Default Claude model
    "gemini": "gemini/gemini-2.0-flash", # Default Gemini model
    "deepseek": "bedrock/us.deepseek.r1-v1:0" # Default Deepseek model
}
# Ensure all mapped models were found
if None in MODEL_MAP.values():
    raise ValueError(f"Could not find a default model in DEFAULT_MODELS for one of the providers: {MODEL_MAP}")

LITELLM_ERROR_PREFIX = "LiteLLM Error:" # Define a prefix for errors generated by our wrapper

# --- State ---
# Store for asynchronous requests
# Results will now store model_id -> text_response_or_error_string
request_store: Dict[str, Dict[str, Any]] = {}

# --- FastMCP Instance ---
mcp = FastMCP(
    title="Multi-Model AI Assistant MCP (LiteLLM)",
    description="A FastMCP server interacting with multiple LLMs via the litellm library.",
    version="0.2.1", # Incremented version for notification feature
)

# --- Helper Functions ---

# New helper function for sending notifications
def send_notification(ctx: Context, request_id: str, status: str):
    """
    Sends a desktop notification on macOS using terminal-notifier.
    Logs a warning if not on macOS or if terminal-notifier fails.
    """
    if platform.system() != "Darwin":
        ctx.debug("Not on macOS (Darwin), skipping terminal-notifier.")
        return

    try:
        title = "MCP Task Complete"
        message = f"Task {request_id} finished with status: {status}"
        # Construct the command
        command = [
            '/Users/minhloi/.rbenv/shims/terminal-notifier',
            '-title', title,
            '-message', message,
            '-sound', 'default', # Optional: adds a sound
            '-group', 'mcp-tasks' # Optional: groups notifications
        ]
        ctx.debug(f"Sending notification: {' '.join(command)}")
        result = subprocess.run(command, check=False, capture_output=True, text=True)

        if result.returncode != 0:
            ctx.warning(
                f"terminal-notifier failed (code {result.returncode}): {result.stderr.strip()}. "
                "Ensure terminal-notifier is installed (`brew install terminal-notifier`)."
            )
    except FileNotFoundError:
        ctx.warning(
            "terminal-notifier command not found. "
            "Ensure it's installed (`brew install terminal-notifier`) and in your PATH."
        )
    except Exception as e:
        ctx.warning(f"Failed to send notification for task {request_id}: {e}")

# New helper function using litellm
def call_litellm(prompt: str, ctx: Context, model: str, system_prompt: Optional[str] = None) -> str:
    """
    Calls the specified LLM using the litellm library. Logs using Context.

    Args:
        prompt: The user prompt.
        ctx: The FastMCP context object for logging.
        model: The specific litellm-compatible model identifier.
        system_prompt: The system prompt to use (optional).

    Returns:
        The text response from the model, or an error message string
        prefixed with LITELLM_ERROR_PREFIX if the call fails.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        ctx.debug(f"Calling litellm.completion for model '{model}'")
        # Set litellm verbosity based on context or a fixed level if desired
        # litellm.set_verbose=ctx.app.debug # Example if FastMCP context had debug flag access

        response = litellm.completion(
            model=model,
            messages=messages
            # You can add other parameters like temperature, max_tokens here
            # e.g., temperature=0.7, max_tokens=500
        )

        # Extract the response content
        # LiteLLM returns a ModelResponse object; we need the message content
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            content = response.choices[0].message.content.strip()
            # Log usage details if needed (available in response.usage)
            usage = response.usage
            ctx.info(f"litellm call successful for model '{model}'. Output length: {len(content)}. Usage: {usage}")
            return content
        else:
            # Handle cases where the response structure is unexpected
            error_message = f"{LITELLM_ERROR_PREFIX} Unexpected response structure from model '{model}': {response}"
            ctx.error(error_message)
            return error_message

    except Exception as e:
        # Catch exceptions raised by litellm (e.g., APIError, RateLimitError, AuthenticationError, etc.)
        error_message = f"{LITELLM_ERROR_PREFIX} calling model '{model}': {type(e).__name__} - {e}"
        ctx.error(error_message)
        return error_message # Return the formatted error string

# Updated background task using call_litellm
def _task_ask_frontier_models(request_id: str, prompt: str, ctx: Context, system_prompt: Optional[str]):
    """
    Background task to query multiple frontier models via litellm.
    Updates the request_store with text results or error messages. Logs using Context.
    """
    results: Dict[str, str] = {} # Store model_id -> text response/error
    status = "completed"
    has_errors = False
    try:
        for model_id in DEFAULT_MODELS:
            ctx.info(f"[{request_id}] Querying model: {model_id}")
            # Pass ctx to call_litellm
            response_text = call_litellm(prompt, ctx, model=model_id, system_prompt=system_prompt)
            results[model_id] = response_text
            # Check if the response text indicates an error from our wrapper
            if response_text.startswith(LITELLM_ERROR_PREFIX):
                ctx.warning(f"[{request_id}] Error querying {model_id}: {response_text}")
                has_errors = True
    except Exception as e:
        ctx.error(f"[{request_id}] Unexpected error during multi-model query loop: {e}")
        status = "error"
        results["_loop_error"] = str(e) # Add overall error if the loop itself fails

    # Adjust final status based on individual model results
    if status == "completed" and has_errors:
        status = "completed_with_errors"
        ctx.warning(f"[{request_id}] Task finished with status: {status}")
    elif status == "error":
        pass # Error already logged
    else:
        ctx.info(f"[{request_id}] Task finished successfully with status: {status}")

    # Update request store
    request_store[request_id]["status"] = status
    request_store[request_id]["results"] = results

    # --- Send notification ---
    # This is called *after* the status is finalized and results are stored.
    send_notification(ctx, request_id, status)
    # --- End notification ---


# --- MCP Tools ---

# Updated to use the background task which now uses litellm
@mcp.tool()
def ask_frontier_models(prompt: str, ctx: Context, system_prompt: Optional[str] = None) -> Dict[str, str]:
    """
    Asynchronously asks a question to all configured frontier models using litellm. Logs using Context.

    Args:
        prompt: The question to ask the models.
        ctx: The FastMCP context object for logging.
        system_prompt: An optional system prompt to guide the models' behavior.

    Returns:
        A dictionary containing the request_id to check the status later.
    """
    request_id = str(uuid.uuid4())
    request_store[request_id] = {"status": "running", "results": {}}
    ctx.info(f"[{request_id}] Starting frontier models task (litellm) for prompt: '{prompt[:50]}...'")

    thread = threading.Thread(
        target=_task_ask_frontier_models,
        args=(request_id, prompt, ctx, system_prompt)
    )
    thread.start()

    return {"request_id": request_id}

# No changes needed here, just reads the store
@mcp.tool()
def check_frontier_models_response(request_id: str, ctx: Context) -> Dict[str, Any]:
    """
    Checks the status and results of a request made via ask_frontier_models. Logs using Context.

    Args:
        request_id: The ID returned by ask_frontier_models.
        ctx: The FastMCP context object for logging.

    Returns:
        A dictionary containing the status ('running', 'completed',
        'completed_with_errors', 'error') and the results (dictionary mapping
        model_id to its text response or error message) if completed.
        Returns an error message if the request_id is not found.
    """
    if request_id not in request_store:
        ctx.error(f"Request ID '{request_id}' not found.")
        return {"status": "error", "error": f"Request ID '{request_id}' not found."}

    ctx.info(f"[{request_id}] Checking status.")
    return request_store[request_id]

# Updated to use call_litellm
@mcp.tool()
def ask_gpt(prompt: str, ctx: Context, system_prompt: Optional[str] = None) -> str:
    """
    Synchronously asks a question to the default GPT model using litellm. Logs using Context.

    Args:
        prompt: The question to ask the model.
        ctx: The FastMCP context object for logging.
        system_prompt: An optional system prompt to guide the model's behavior.

    Returns:
        The raw text response (or error message prefixed with LITELLM_ERROR_PREFIX) from litellm for the GPT model.
    """
    model_id = MODEL_MAP["gpt"]
    if not model_id: # Should not happen due to check at start, but good practice
         error_msg = f"{LITELLM_ERROR_PREFIX} No default GPT model configured."
         ctx.error(error_msg)
         return error_msg
    ctx.info(f"Asking GPT model ({model_id}) synchronously via litellm: '{prompt[:50]}...'")
    return call_litellm(prompt, ctx, model=model_id, system_prompt=system_prompt)

# Updated to use call_litellm
@mcp.tool()
def ask_claude(prompt: str, ctx: Context, system_prompt: Optional[str] = None) -> str:
    """
    Synchronously asks a question to the default Claude model using litellm. Logs using Context.

    Args:
        prompt: The question to ask the model.
        ctx: The FastMCP context object for logging.
        system_prompt: An optional system prompt to guide the model's behavior.

    Returns:
        The raw text response (or error message prefixed with LITELLM_ERROR_PREFIX) from litellm for the Claude model.
    """
    model_id = MODEL_MAP["claude"]
    if not model_id:
         error_msg = f"{LITELLM_ERROR_PREFIX} No default Claude model configured."
         ctx.error(error_msg)
         return error_msg
    ctx.info(f"Asking Claude model ({model_id}) synchronously via litellm: '{prompt[:50]}...'")
    return call_litellm(prompt, ctx, model=model_id, system_prompt=system_prompt)

# Updated to use call_litellm
@mcp.tool()
def ask_gemini(prompt: str, ctx: Context, system_prompt: Optional[str] = None) -> str:
    """
    Synchronously asks a question to the default Gemini model using litellm. Logs using Context.

    Args:
        prompt: The question to ask the model.
        ctx: The FastMCP context object for logging.
        system_prompt: An optional system prompt to guide the model's behavior.

    Returns:
        The raw text response (or error message prefixed with LITELLM_ERROR_PREFIX) from litellm for the Gemini model.
    """
    model_id = MODEL_MAP["gemini"]
    if not model_id:
         error_msg = f"{LITELLM_ERROR_PREFIX} No default Gemini model configured."
         ctx.error(error_msg)
         return error_msg
    ctx.info(f"Asking Gemini model ({model_id}) synchronously via litellm: '{prompt[:50]}...'")
    return call_litellm(prompt, ctx, model=model_id, system_prompt=system_prompt)

# Updated to use call_litellm
@mcp.tool()
def ask_deepseek(prompt: str, ctx: Context, system_prompt: Optional[str] = None) -> str:
    """
    Synchronously asks a question to the default Deepseek model using litellm. Logs using Context.

    Args:
        prompt: The question to ask the model.
        ctx: The FastMCP context object for logging.
        system_prompt: An optional system prompt to guide the model's behavior.

    Returns:
        The raw text response (or error message prefixed with LITELLM_ERROR_PREFIX) from litellm for the Deepseek model.
    """
    model_id = MODEL_MAP["deepseek"]
    if not model_id:
         error_msg = f"{LITELLM_ERROR_PREFIX} No default Deepseek model configured."
         ctx.error(error_msg)
         return error_msg
    ctx.info(f"Asking Deepseek model ({model_id}) synchronously via litellm: '{prompt[:50]}...'")
    return call_litellm(prompt, ctx, model=model_id, system_prompt=system_prompt)

if __name__ == "__main__":
    # Configure API keys for litellm if needed (e.g., via environment variables)
    # Example: os.environ["OPENAI_API_KEY"] = "your_key"
    # Example: os.environ["ANTHROPIC_API_KEY"] = "your_key"
    # LiteLLM automatically picks up keys from common environment variables.
    print("Starting FastMCP server (using LiteLLM)...")
    mcp.run()
