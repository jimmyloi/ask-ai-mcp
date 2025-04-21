import uuid
import subprocess
import threading
import json
from typing import List, Dict, Any, Optional

from fastmcp import FastMCP

# --- Configuration ---
# Adjust these model names if your aichat configuration uses different identifiers
DEFAULT_MODELS = [
    "openai:o1",
    "claude:claude-3-7-sonnet-20250219",
    "gemini:gemini-2.0-flash",
    "deepseek:bedrock_deepseek_r1"
]
# Ensure 'aichat' command is in the system's PATH
AICHAT_COMMAND = "aichat"

# --- In-memory store for async requests ---
# Warning: This is not persistent. Server restarts will lose state.
# For production, consider using a database or persistent cache.
request_store: Dict[str, Dict[str, Any]] = {}

# --- Initialize FastMCP ---
mcp = FastMCP(
    title="Multi-Model AI Assistant MCP",
    description="A FastMCP server interacting with multiple LLMs via aichat CLI.",
    version="0.1.0",
)

# --- Helper Function to run aichat ---
def run_aichat(prompt: str, model: Optional[str] = None, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Runs the aichat command and returns the result."""
    command = [AICHAT_COMMAND]
    if model:
        command.extend(["--model", model])
    if system_prompt:
        # Ensure system prompt is passed correctly, potentially needing quotes handled by subprocess
        command.extend(["--system", system_prompt])

    command.append(prompt)

    try:
        print(f"Running command: {' '.join(command)}") # For debugging
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True, # Raise an exception if aichat returns non-zero exit code
            encoding='utf-8'
        )
        return {"success": True, "output": result.stdout.strip()}
    except FileNotFoundError:
        print(f"Error: '{AICHAT_COMMAND}' command not found. Is aichat installed and in PATH?")
        return {"success": False, "error": f"'{AICHAT_COMMAND}' not found."}
    except subprocess.CalledProcessError as e:
        print(f"Error running aichat: {e}")
        print(f"Stderr: {e.stderr}")
        return {"success": False, "error": e.stderr.strip() or f"aichat exited with status {e.returncode}"}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"success": False, "error": str(e)}

# --- Background Task Functions ---

def _task_ask_multimodels(request_id: str, prompt: str, models: List[str]):
    """Background task for ask_multimodels."""
    results = {}
    status = "completed"
    for model_name in models:
        print(f"[{request_id}] Querying model: {model_name}")
        result = run_aichat(prompt=prompt, model=model_name)
        results[model_name] = result
        if not result["success"]:
            status = "completed_with_errors" # Mark if any model failed
            print(f"[{request_id}] Error querying {model_name}: {result.get('error')}")
        else:
             print(f"[{request_id}] Received response from {model_name}")

    # Update the store
    request_store[request_id]["status"] = status
    request_store[request_id]["results"] = results
    print(f"[{request_id}] Multimodel request finished with status: {status}")

def _task_ask_model_with_multi_personas(request_id: str, prompt: str, model: str, system_prompts: List[str]):
    """Background task for ask_model_with_multi_personas."""
    results = {}
    status = "completed"
    for i, sys_prompt in enumerate(system_prompts):
        persona_key = f"persona_{i+1}"
        print(f"[{request_id}] Querying model {model} with {persona_key}")
        result = run_aichat(prompt=prompt, model=model, system_prompt=sys_prompt)
        results[persona_key] = {
            "system_prompt": sys_prompt,
            "result": result
        }
        if not result["success"]:
            status = "completed_with_errors"
            print(f"[{request_id}] Error querying {model} with {persona_key}: {result.get('error')}")
        else:
            print(f"[{request_id}] Received response from {model} with {persona_key}")


    # Update the store
    request_store[request_id]["status"] = status
    request_store[request_id]["results"] = results
    print(f"[{request_id}] Multi-persona request finished with status: {status}")


# --- MCP Tools ---

@mcp.tool()
def ask_multimodels(prompt: str, models: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Asynchronously asks a question to multiple LLM models via the aichat CLI.

    Args:
        prompt: The question/prompt to ask the models.
        models: A list of model identifiers (as configured in aichat) to query.
                If None, uses default models.

    Returns:
        A dictionary containing the request_id and the initial status ('submitted').
    """
    request_id = str(uuid.uuid4())
    models_to_use = models if models else DEFAULT_MODELS

    # Initialize status in the store
    request_store[request_id] = {
        "status": "pending",
        "prompt": prompt,
        "models_queried": models_to_use,
        "results": {}
    }

    # Start background thread
    thread = threading.Thread(
        target=_task_ask_multimodels,
        args=(request_id, prompt, models_to_use),
        daemon=True # Allows main program to exit even if threads are running
    )
    thread.start()

    print(f"Submitted ask_multimodels request: {request_id}")
    return {"request_id": request_id, "status": "submitted"}

@mcp.tool()
def check_multimodel_request(request_id: str) -> Dict[str, Any]:
    """
    Checks the status and results of a request made with ask_multimodels.

    Args:
        request_id: The ID of the request to check.

    Returns:
        A dictionary containing the status and results (if available)
        or an error message if the request ID is not found.
    """
    if request_id in request_store:
        print(f"Checking status for request: {request_id}")
        return request_store[request_id]
    else:
        print(f"Request ID not found: {request_id}")
        return {"error": "Request ID not found", "request_id": request_id}

@mcp.tool()
def ask_model(prompt: str, model: str) -> Dict[str, Any]:
    """
    Synchronously asks a question to a specific LLM model via the aichat CLI.

    Args:
        prompt: The question/prompt to ask the model.
        model: The model identifier (as configured in aichat) to query.
               Must be one of the supported models.

    Returns:
        A dictionary containing the result ('output') or an error message.
    """
    # Optional: Add validation for the model name if needed
    # if model not in DEFAULT_MODELS: # Or a broader list of allowed models
    #     return {"success": False, "error": f"Model '{model}' is not supported by this tool."}

    print(f"Asking model '{model}' synchronously...")
    result = run_aichat(prompt=prompt, model=model)
    print(f"Received synchronous response from '{model}'. Success: {result['success']}")
    return result # Contains 'success' and either 'output' or 'error'

@mcp.tool()
def ask_model_with_multi_personas(
    prompt: str,
    model: str,
    system_prompts: List[str]
) -> Dict[str, str]:
    """
    Asynchronously asks a question to a specific model using multiple system prompts (personas).

    Args:
        prompt: The question/prompt to ask the model.
        model: The model identifier (as configured in aichat).
        system_prompts: A list of system prompts to use as different personas.

    Returns:
        A dictionary containing the request_id and the initial status ('submitted').
    """
    request_id = str(uuid.uuid4())

    # Initialize status in the store
    request_store[request_id] = {
        "status": "pending",
        "prompt": prompt,
        "model": model,
        "system_prompts": system_prompts,
        "results": {}
    }

    # Start background thread
    thread = threading.Thread(
        target=_task_ask_model_with_multi_personas,
        args=(request_id, prompt, model, system_prompts),
        daemon=True
    )
    thread.start()

    print(f"Submitted ask_model_with_multi_personas request: {request_id}")
    return {"request_id": request_id, "status": "submitted"}

@mcp.tool()
def check_model_with_multi_personas_request(request_id: str) -> Dict[str, Any]:
    """
    Checks the status and results of a request made with ask_model_with_multi_personas.

    Args:
        request_id: The ID of the request to check.

    Returns:
        A dictionary containing the status and results (if available)
        or an error message if the request ID is not found.
    """
    # This function is identical to check_multimodel_request, just for the other async tool
    if request_id in request_store:
        print(f"Checking status for multi-persona request: {request_id}")
        return request_store[request_id]
    else:
        print(f"Multi-persona request ID not found: {request_id}")
        return {"error": "Request ID not found", "request_id": request_id}


# --- Run the server ---
if __name__ == "__main__":
    print("Starting FastMCP server...")
    print(f"Default models: {DEFAULT_MODELS}")
    print(f"Using aichat command: {AICHAT_COMMAND}")
    print("Make sure 'aichat' is installed, configured, and in your PATH.")
    # You might want to configure host and port here
    # mcp.run(host="0.0.0.0", port=8000)
    mcp.run() # Runs on default host/port (usually 127.0.0.1:8000)
