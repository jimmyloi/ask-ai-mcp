import uuid
import subprocess
import threading
import json
import shlex
from typing import List, Dict, Any, Optional

from fastmcp import FastMCP

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


AICHAT_COMMAND = "aichat"

# --- State ---
# Store for asynchronous requests
request_store: Dict[str, Dict[str, Any]] = {}

# --- FastMCP Instance ---
mcp = FastMCP(
    title="Multi-Model AI Assistant MCP",
    description="A FastMCP server interacting with multiple LLMs via aichat CLI.",
    version="0.1.0",
    # Enable background tasks if FastMCP supports it directly,
    # otherwise threading is used explicitly below.
    # background_tasks=True # Example if FastMCP has built-in support
)

# --- Helper Functions ---

def run_aichat(prompt: str, model: Optional[str] = None, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Runs the aichat CLI command and returns the parsed JSON output.

    Args:
        prompt: The user prompt.
        model: The specific model to use (optional).
        system_prompt: The system prompt to use (optional).

    Returns:
        A dictionary containing the response from aichat.

    Raises:
        RuntimeError: If the aichat command fails or returns invalid JSON.
    """
    cmd = [AICHAT_COMMAND, "--output-json"]
    if model:
        cmd.extend(["--model", model])
    if system_prompt:
        # Aichat uses -s or --system for system prompt
        cmd.extend(["-s", system_prompt])

    cmd.append(prompt)

    try:
        print(f"Running command: {' '.join(shlex.quote(c) for c in cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Try to parse the last line as JSON, as aichat might print status/info first
        json_output = None
        for line in reversed(result.stdout.strip().splitlines()):
             try:
                 json_output = json.loads(line)
                 break
             except json.JSONDecodeError:
                 continue # Ignore lines that are not valid JSON

        if json_output is None:
             raise ValueError("No valid JSON output found from aichat.")

        print(f"Command successful. Output keys: {json_output.keys()}")
        return json_output

    except subprocess.CalledProcessError as e:
        error_message = f"aichat command failed with exit code {e.returncode}.\nStderr: {e.stderr}\nStdout: {e.stdout}"
        print(error_message)
        # Return an error structure consistent with successful runs if possible
        return {"error": error_message, "status": "error"}
    except FileNotFoundError:
        error_message = f"Error: '{AICHAT_COMMAND}' command not found. Make sure aichat is installed and in your PATH."
        print(error_message)
        return {"error": error_message, "status": "error"}
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        print(error_message)
        return {"error": error_message, "status": "error"}


def _task_ask_frontier_models(request_id: str, prompt: str, system_prompt: Optional[str]):
    """
    Background task to query multiple frontier models via aichat.
    Updates the request_store with results.
    """
    results = {}
    status = "completed"
    try:
        for model_id in DEFAULT_MODELS:
            print(f"[{request_id}] Querying model: {model_id}")
            response = run_aichat(prompt, model=model_id, system_prompt=system_prompt)
            results[model_id] = response
            if response.get("error"):
                print(f"[{request_id}] Error querying {model_id}: {response['error']}")
                # Optionally set overall status to partial_error or keep completed
                # status = "partial_error" # Or similar if needed
    except Exception as e:
        print(f"[{request_id}] Unexpected error during multi-model query: {e}")
        status = "error"
        results["error"] = str(e) # Add overall error if the loop fails

    request_store[request_id]["status"] = status
    request_store[request_id]["results"] = results
    print(f"[{request_id}] Task finished with status: {status}")


# --- MCP Tools ---

@mcp.tool()
def ask_frontier_models(prompt: str, system_prompt: Optional[str] = None) -> Dict[str, str]:
    """
    Asynchronously asks a question to all configured frontier models.

    Args:
        prompt: The question to ask the models.
        system_prompt: An optional system prompt to guide the models' behavior.

    Returns:
        A dictionary containing the request_id to check the status later.
    """
    request_id = str(uuid.uuid4())
    request_store[request_id] = {"status": "running", "results": {}}
    print(f"[{request_id}] Starting frontier models task for prompt: '{prompt[:50]}...'")

    # Run the actual querying in a background thread
    thread = threading.Thread(
        target=_task_ask_frontier_models,
        args=(request_id, prompt, system_prompt)
    )
    thread.start()

    return {"request_id": request_id}

@mcp.tool()
def check_frontier_models_response(request_id: str) -> Dict[str, Any]:
    """
    Checks the status and results of a request made via ask_frontier_models.

    Args:
        request_id: The ID returned by ask_frontier_models.

    Returns:
        A dictionary containing the status ('running', 'completed', 'error')
        and the results from the models if completed. Returns an error
        message if the request_id is not found.
    """
    if request_id not in request_store:
        return {"error": f"Request ID '{request_id}' not found."}

    print(f"[{request_id}] Checking status.")
    return request_store[request_id]

@mcp.tool()
def ask_gpt(prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Synchronously asks a question to the default GPT model.

    Args:
        prompt: The question to ask the model.
        system_prompt: An optional system prompt to guide the model's behavior.

    Returns:
        The response dictionary directly from the aichat command for the GPT model.
    """
    model_id = MODEL_MAP["gpt"]
    print(f"Asking GPT model ({model_id}) synchronously: '{prompt[:50]}...'")
    return run_aichat(prompt, model=model_id, system_prompt=system_prompt)

@mcp.tool()
def ask_claude(prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Synchronously asks a question to the default Claude model.

    Args:
        prompt: The question to ask the model.
        system_prompt: An optional system prompt to guide the model's behavior.

    Returns:
        The response dictionary directly from the aichat command for the Claude model.
    """
    model_id = MODEL_MAP["claude"]
    print(f"Asking Claude model ({model_id}) synchronously: '{prompt[:50]}...'")
    return run_aichat(prompt, model=model_id, system_prompt=system_prompt)

@mcp.tool()
def ask_gemini(prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Synchronously asks a question to the default Gemini model.

    Args:
        prompt: The question to ask the model.
        system_prompt: An optional system prompt to guide the model's behavior.

    Returns:
        The response dictionary directly from the aichat command for the Gemini model.
    """
    model_id = MODEL_MAP["gemini"]
    print(f"Asking Gemini model ({model_id}) synchronously: '{prompt[:50]}...'")
    return run_aichat(prompt, model=model_id, system_prompt=system_prompt)

@mcp.tool()
def ask_deepseek(prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Synchronously asks a question to the default Deepseek model.

    Args:
        prompt: The question to ask the model.
        system_prompt: An optional system prompt to guide the model's behavior.

    Returns:
        The response dictionary directly from the aichat command for the Deepseek model.
    """
    model_id = MODEL_MAP["deepseek"]
    print(f"Asking Deepseek model ({model_id}) synchronously: '{prompt[:50]}...'")
    return run_aichat(prompt, model=model_id, system_prompt=system_prompt)

if __name__ == "__main__":
    mcp.run()
