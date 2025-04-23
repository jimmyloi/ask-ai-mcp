# Multi-Model AI Assistant MCP (LiteLLM)

Version: 0.2.1

## Description

This project provides a FastMCP server that acts as a Master Control Program (MCP) to interact with various Large Language Models (LLMs). It leverages the [LiteLLM library](https://github.com/BerriAI/litellm) to provide a unified interface for querying different model providers like OpenAI, Anthropic, Google (Gemini), and AWS Bedrock.

The server exposes tools to query specific models directly or to query a set of default "frontier" models concurrently and retrieve their responses asynchronously.

## Features

*   Interact with multiple LLMs through a single interface.
*   Query specific model families (GPT, Claude, Gemini, Deepseek) via dedicated tools.
*   Asynchronously query a predefined list of "frontier" models simultaneously.
*   Check the status and retrieve results from asynchronous multi-model queries.
*   Uses LiteLLM for broad LLM provider compatibility.
*   Built with FastMCP for easy tool definition and execution.
*   Basic notification mechanism for asynchronous task completion.

## Prerequisites

*   Python 3.x
*   Access keys/API keys for the desired LLM providers (e.g., OpenAI, Anthropic, Google AI Studio, AWS).

## Installation

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Install dependencies:**
    Ensure `fastmcp` and `litellm` are listed in your `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **LiteLLM API Keys:**
    LiteLLM requires API keys to be set as environment variables. Set the necessary environment variables for the providers you intend to use. Refer to the [LiteLLM Documentation on Providers](https://docs.litellm.ai/docs/providers) for the specific environment variable names (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION_NAME`).
    ```bash
    # Example for bash/zsh
    export OPENAI_API_KEY="your_openai_key"
    export ANTHROPIC_API_KEY="your_anthropic_key"
    # ... add others as needed
    ```

2.  **Model Selection (Optional):**
    The default models used by the server are defined in `mcp_server.py`:
    *   `DEFAULT_MODELS`: A list of models queried by `ask_frontier_models`.
    *   `MODEL_MAP`: A dictionary mapping short names (like "gpt", "claude") to specific LiteLLM model strings used by tools like `ask_gpt`.

    Review and update these lists/mappings in `mcp_server.py` according to the models you have access to and wish to use. The current entries are examples and might need changing.

## Running the Server

Ensure your environment variables (API keys) are set correctly in your terminal session. Then, run the FastMCP server:

```bash
fastmcp serve mcp_server:mcp
```

This will start the server, typically on `http://127.0.0.1:8000` (check the FastMCP output for the exact address). You can then interact with it using a FastMCP client or via its web interface (usually at `/docs`).

## Available Tools (MCP Functions)

You can call these functions using a FastMCP client connected to the running server.

*   **`ask_frontier_models(prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]`**
    *   Initiates *asynchronous* requests to all models listed in `DEFAULT_MODELS`.
    *   **Parameters:**
        *   `prompt` (str): The user's prompt for the models.
        *   `system_prompt` (Optional[str]): An optional system message or context.
    *   **Returns:** A dictionary containing a `request_id` (str) which you need to use to check the status later. Example: `{"request_id": "some-unique-id"}`

*   **`check_frontier_models_response(request_id: str) -> Dict[str, Any]`**
    *   Checks the status and retrieves results for a previously initiated `ask_frontier_models` request.
    *   **Parameters:**
        *   `request_id` (str): The ID returned by `ask_frontier_models`.
    *   **Returns:** A dictionary containing the `status` ("running", "completed", "completed_with_errors", "error") and, if completed, the `results` (a dictionary mapping model names to their responses or error messages).
        *   Example (Running): `{"status": "running", "results": {}}`
        *   Example (Completed): `{"status": "completed", "results": {"openai/o1": "Response from OpenAI...", "anthropic/claude-3-5-sonnet-20240620": "Response from Anthropic...", "gemini/gemini-2.0-flash": "LiteLLM Error: AuthenticationError...", ...}}`
        *   Example (Error): `{"status": "error", "error": "Request ID '...' not found."}`

*   **`ask_gpt(prompt: str, system_prompt: Optional[str] = None) -> str`**
    *   Sends a *synchronous* request to the default GPT model specified in `MODEL_MAP["gpt"]`.
    *   **Parameters:**
        *   `prompt` (str): The user's prompt.
        *   `system_prompt` (Optional[str]): An optional system message.
    *   **Returns:** The response string from the model, or an error message prefixed with `LiteLLM Error:`.

*   **`ask_claude(prompt: str, system_prompt: Optional[str] = None) -> str`**
    *   Sends a *synchronous* request to the default Claude model specified in `MODEL_MAP["claude"]`.
    *   Parameters and Returns are similar to `ask_gpt`.

*   **`ask_gemini(prompt: str, system_prompt: Optional[str] = None) -> str`**
    *   Sends a *synchronous* request to the default Gemini model specified in `MODEL_MAP["gemini"]`.
    *   Parameters and Returns are similar to `ask_gpt`.

*   **`ask_deepseek(prompt: str, system_prompt: Optional[str] = None) -> str`**
    *   Sends a *synchronous* request to the default Deepseek model specified in `MODEL_MAP["deepseek"]`.
    *   Parameters and Returns are similar to `ask_gpt`.

## Example Workflow (Asynchronous Frontier Models)

1.  **Initiate Request:** Call `ask_frontier_models` with your prompt.
    ```python
    # Example using a hypothetical FastMCP client
    response = client.call("ask_frontier_models", prompt="Explain the theory of relativity simply.")
    request_id = response["request_id"]
    print(f"Request initiated with ID: {request_id}")
    ```
2.  **Check Status:** Periodically call `check_frontier_models_response` with the `request_id`.
    ```python
    import time
    # Example using a hypothetical FastMCP client
    while True:
        status_response = client.call("check_frontier_models_response", request_id=request_id)
        status = status_response["status"]
        print(f"Current status: {status}")
        if status in ["completed", "completed_with_errors", "error"]:
            print("Results:")
            print(status_response.get("results", "No results available."))
            if status == "error":
                print(f"Error details: {status_response.get('error', 'Unknown error')}")
            break
        elif status == "running":
            time.sleep(5) # Wait before checking again
        else:
            print(f"Unknown status: {status}")
            break

    ```

## Error Handling

Errors encountered during LLM calls via LiteLLM will typically be returned as strings prefixed with `LiteLLM Error:` (as defined by `LITELLM_ERROR_PREFIX` in the code). Check the response strings from the tools or the `results` dictionary from `check_frontier_models_response`. Errors during the asynchronous task loop or finding the request ID will be reported in the `error` field of the `check_frontier_models_response` output.
