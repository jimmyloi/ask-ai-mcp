import uuid
import subprocess
import threading
import json # Keep json for potential future use or other parts, but not for aichat output
import shlex
from typing import List, Dict, Any, Optional

# Import Context from fastmcp
from fastmcp import FastMCP, Context

# --- Configuration ---
# Assuming these are the "frontier" models
DEFAULT_MODELS = [
    "openai:o1",
    "claude:claude-3-7-sonnet-20250219",
    "gemini:gemini-2.0-flash",
    "deepseek:bedrock_deepseek_r1"
]
# Map tool names to specific model identifiers from DEFAULT_MODELS
# We'll use the first one found for each provider as a default for the specific tools
MODEL_MAP = {
    "gpt": next((m for m in DEFAULT_MODELS if m.startswith("openai:")), None),
    "claude": next((m for m in DEFAULT_MODELS if m.startswith("claude:")), None),
    "gemini": next((m for m in DEFAULT_MODELS if m.startswith("gemini:")), None),
    "deepseek": next((m for m in DEFAULT_MODELS if m.startswith("deepseek:")), None),
}
# Ensure all mapped models were found
if None in MODEL_MAP.values():
    raise ValueError(f"Could not find a default model in DEFAULT_MODELS for one of the providers: {MODEL_MAP}")

AICHAT_COMMAND = "/opt/homebrew/bin/aichat"
# Define prefixes to identify error messages returned by run_aichat
AICHAT_ERROR_PREFIXES = ("aichat command failed", "Error:", "An unexpected error occurred")


# --- State ---
# Store for asynchronous requests
# Results will now store model_id -> text_response_or_error_string
request_store: Dict[str, Dict[str, Any]] = {}

# --- FastMCP Instance ---
mcp = FastMCP(
    title="Multi-Model AI Assistant MCP",
    description="A FastMCP server interacting with multiple LLMs via aichat CLI.",
    version="0.1.0",
)

# --- Helper Functions ---

# Add ctx: Context argument
def run_aichat(prompt: str, model: Optional[str] = None, system_prompt: Optional[str] = None, ctx: Context) -> str:
    """
    Runs the aichat CLI command and returns the text output. Logs using Context.

    Args:
        prompt: The user prompt.
        model: The specific model to use (optional).
        system_prompt: The system prompt to use (optional).
        ctx: The FastMCP context object for logging.

    Returns:
        The raw text output from aichat, or an error message string
        if the command fails.
    """
    cmd = [AICHAT_COMMAND] # Removed --output-json
    if model:
        cmd.extend(["--model", model])
    if system_prompt:
        # Aichat uses -s or --system for system prompt
        cmd.extend(["--prompt", system_prompt])

    cmd.append(prompt)

    try:
        # Use ctx.debug for detailed command logging
        cmd_str = ' '.join(shlex.quote(c) for c in cmd)
        ctx.debug(f"Running command: {cmd_str}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8') # Specify encoding
        output_text = result.stdout.strip()
        # Use ctx.info for successful execution summary
        ctx.info(f"Command successful for model '{model or 'default'}'. Output length: {len(output_text)}")
        return output_text

    except subprocess.CalledProcessError as e:
        # Combine stdout and stderr for better error context
        error_message = (
            f"aichat command failed with exit code {e.returncode}.\n"
            f"Stderr: {e.stderr.strip()}\n"
            f"Stdout: {e.stdout.strip()}"
        )
        # Use ctx.error for command failures
        ctx.error(f"Error running command for model '{model or 'default'}': {error_message}")
        return error_message # Return the error message string
    except FileNotFoundError:
        error_message = f"Error: '{AICHAT_COMMAND}' command not found. Make sure aichat is installed and in your PATH."
        # Use ctx.error for environment issues
        ctx.error(error_message)
        return error_message # Return the error message string
    except Exception as e:
        error_message = f"An unexpected error occurred during aichat execution: {e}"
        # Use ctx.error for unexpected exceptions
        ctx.error(error_message)
        return error_message # Return the error message string

# Add ctx: Context argument
def _task_ask_frontier_models(request_id: str, prompt: str, system_prompt: Optional[str], ctx: Context):
    """
    Background task to query multiple frontier models via aichat.
    Updates the request_store with text results or error messages. Logs using Context.
    """
    results: Dict[str, str] = {} # Store model_id -> text response/error
    status = "completed"
    has_errors = False
    try:
        for model_id in DEFAULT_MODELS:
            # Use ctx.info for progress within the task
            ctx.info(f"[{request_id}] Querying model: {model_id}")
            # Pass ctx to run_aichat
            response_text = run_aichat(prompt, model=model_id, system_prompt=system_prompt, ctx=ctx)
            results[model_id] = response_text
            # Check if the response text indicates an error
            if response_text.startswith(AICHAT_ERROR_PREFIXES):
                # Use ctx.warning for individual model errors within the task
                ctx.warning(f"[{request_id}] Error querying {model_id}: {response_text}")
                has_errors = True
    except Exception as e:
        # Use ctx.error for unexpected errors during the loop itself
        ctx.error(f"[{request_id}] Unexpected error during multi-model query loop: {e}")
        status = "error"
        results["_loop_error"] = str(e) # Add overall error if the loop itself fails

    # Adjust final status based on individual model results
    if status == "completed" and has_errors:
        status = "completed_with_errors"
        # Use ctx.warning if the task completed but had some errors
        ctx.warning(f"[{request_id}] Task finished with status: {status}")
    elif status == "error":
        # ctx.error was already called above if status is 'error'
        pass
    else:
        # Use ctx.info for successful completion
        ctx.info(f"[{request_id}] Task finished successfully with status: {status}")


    # Update request store (this part doesn't need context logging itself)
    request_store[request_id]["status"] = status
    request_store[request_id]["results"] = results


# --- MCP Tools ---

# Add ctx: Context argument
@mcp.tool()
def ask_frontier_models(prompt: str, system_prompt: Optional[str] = None, ctx: Context) -> Dict[str, str]:
    """
    Asynchronously asks a question to all configured frontier models. Logs using Context.

    Args:
        prompt: The question to ask the models.
        system_prompt: An optional system prompt to guide the models' behavior.
        ctx: The FastMCP context object for logging.

    Returns:
        A dictionary containing the request_id to check the status later.
    """
    request_id = str(uuid.uuid4())
    # Results will store text strings now
    request_store[request_id] = {"status": "running", "results": {}}
    # Use ctx.info to log the start of the task
    ctx.info(f"[{request_id}] Starting frontier models task for prompt: '{prompt[:50]}...'")

    # Run the actual querying in a background thread
    # Pass ctx to the target function
    thread = threading.Thread(
        target=_task_ask_frontier_models,
        args=(request_id, prompt, system_prompt, ctx) # Pass ctx here
    )
    thread.start()

    return {"request_id": request_id}

# Add ctx: Context argument
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
        # Use ctx.error for invalid request IDs
        ctx.error(f"Request ID '{request_id}' not found.")
        # Return structure consistent with successful check but indicating the ID error
        return {"status": "error", "error": f"Request ID '{request_id}' not found."}

    # Use ctx.info for status check action
    ctx.info(f"[{request_id}] Checking status.")
    # Return the whole entry, which now contains text results
    return request_store[request_id]

# Add ctx: Context argument
@mcp.tool()
def ask_gpt(prompt: str, system_prompt: Optional[str] = None, ctx: Context) -> str:
    """
    Synchronously asks a question to the default GPT model. Logs using Context.

    Args:
        prompt: The question to ask the model.
        system_prompt: An optional system prompt to guide the model's behavior.
        ctx: The FastMCP context object for logging.
        ctx: The FastMCP context object for logging.
        ctx: The FastMCP context object for logging.
        ctx: The FastMCP context object for logging.

    Returns:
        The raw text response (or error message) from the aichat command for the GPT model.
    """
    model_id = MODEL_MAP["gpt"]
    # Use ctx.info for starting the synchronous call
    ctx.info(f"Asking GPT model ({model_id}) synchronously: '{prompt[:50]}...'")
    # Pass ctx to run_aichat
    return run_aichat(prompt, model=model_id, system_prompt=system_prompt, ctx=ctx)

# Add ctx: Context argument
@mcp.tool()
def ask_claude(prompt: str, system_prompt: Optional[str] = None, ctx: Context) -> str:
    """
    Synchronously asks a question to the default Claude model. Logs using Context.

    Args:
        prompt: The question to ask the model.
        system_prompt: An optional system prompt to guide the model's behavior.

    Returns:
        The raw text response (or error message) from the aichat command for the Claude model.
    """
    model_id = MODEL_MAP["claude"]
    # Use ctx.info for starting the synchronous call
    ctx.info(f"Asking Claude model ({model_id}) synchronously: '{prompt[:50]}...'")
    # Pass ctx to run_aichat
    return run_aichat(prompt, model=model_id, system_prompt=system_prompt, ctx=ctx)

# Add ctx: Context argument
@mcp.tool()
def ask_gemini(prompt: str, system_prompt: Optional[str] = None, ctx: Context) -> str:
    """
    Synchronously asks a question to the default Gemini model. Logs using Context.

    Args:
        prompt: The question to ask the model.
        system_prompt: An optional system prompt to guide the model's behavior.

    Returns:
        The raw text response (or error message) from the aichat command for the Gemini model.
    """
    model_id = MODEL_MAP["gemini"]
    # Use ctx.info for starting the synchronous call
    ctx.info(f"Asking Gemini model ({model_id}) synchronously: '{prompt[:50]}...'")
    # Pass ctx to run_aichat
    return run_aichat(prompt, model=model_id, system_prompt=system_prompt, ctx=ctx)

# Add ctx: Context argument
@mcp.tool()
def ask_deepseek(prompt: str, system_prompt: Optional[str] = None, ctx: Context) -> str:
    """
    Synchronously asks a question to the default Deepseek model. Logs using Context.

    Args:
        prompt: The question to ask the model.
        system_prompt: An optional system prompt to guide the model's behavior.

    Returns:
        The raw text response (or error message) from the aichat command for the Deepseek model.
    """
    model_id = MODEL_MAP["deepseek"]
    # Use ctx.info for starting the synchronous call
    ctx.info(f"Asking Deepseek model ({model_id}) synchronously: '{prompt[:50]}...'")
    # Pass ctx to run_aichat
    return run_aichat(prompt, model=model_id, system_prompt=system_prompt, ctx=ctx)

if __name__ == "__main__":
    # No context available here, standard print is fine or use Python's logging module
    print("Starting FastMCP server...")
    mcp.run()
