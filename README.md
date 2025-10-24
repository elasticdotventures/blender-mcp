

# BlenderMCP - Blender Model Context Protocol Integration

BlenderMCP connects Blender to Claude AI through the Model Context Protocol (MCP), allowing Claude to directly interact with and control Blender. This integration enables prompt assisted 3D modeling, scene creation, and manipulation.

**We have no official website. Any website you see online is unofficial and has no affiliation with this project. Use them at your own risk.**

[Full tutorial](https://www.youtube.com/watch?v=lCyQ717DuzQ)

### Join the Community

Give feedback, get inspired, and build on top of the MCP: [Discord](https://discord.gg/z5apgR8TFU)

### Supporters

[CodeRabbit](https://www.coderabbit.ai/)

[Satish Goda](https://github.com/satishgoda)

**All supporters:**

[Support this project](https://github.com/sponsors/ahujasid)

## Release notes (1.3.0)
- Vision via vLLM: inspect single or multiple Blender views and images using OpenAI-compatible vLLM endpoints (e.g., DeepSeek-OCR)
- Chains: register/update self-defined (MCP-addressable & mutable) LangChain v1-like specs, including per-chain filters and views
- Multi-model runs: pass multiple models per request or per chain and aggregate results
- Scene filters: basic boolean filters (collision/material) evaluated in Blender alongside text filters
- Map-reduce groundwork: simple concatenation aggregator for multi-view and multi-model responses


### Previously added features:
- Support for Poly Haven assets through their API
- Support to generate 3D models using Hyper3D Rodin
- For newcomers, you can go straight to Installation. For existing users, see the points below
- Download the latest addon.py file and replace the older one, then add it to Blender
- Delete the MCP server from Claude and add it back again, and you should be good to go!

## Features

- **Two-way communication**: Connect Claude AI to Blender through a socket-based server
- **Object manipulation**: Create, modify, and delete 3D objects in Blender
- **Material control**: Apply and modify materials and colors
- **Scene inspection**: Get detailed information about the current Blender scene
- **Code execution**: Run arbitrary Python code in Blender from Claude
- **Vision analysis**: Query vLLM-hosted vision models with chain-defined prompts, views, and filters

## Components

The system consists of two main components:

1. **Blender Addon (`addon.py`)**: A Blender addon that creates a socket server within Blender to receive and execute commands
2. **MCP Server (`src/blender_mcp/server.py`)**: A Python server that implements the Model Context Protocol and connects to the Blender addon

## Installation


### Prerequisites

- Blender 3.0 or newer
- Python 3.10 or newer
- uv package manager: 

**If you're on Mac, please install uv as**
```bash
brew install uv
```
**On Windows**
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex" 
```
and then
```bash
set Path=C:\Users\nntra\.local\bin;%Path%
```

Otherwise installation instructions are on their website: [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

**⚠️ Do not proceed before installing UV**

### Environment Variables

The following environment variables can be used to configure the Blender connection:

- `BLENDER_HOST`: Host address for Blender socket server (default: "localhost")
- `BLENDER_PORT`: Port number for Blender socket server (default: 9876)

Example:
```bash
export BLENDER_HOST='host.docker.internal'
export BLENDER_PORT=9876
```

### Claude for Desktop Integration

[Watch the setup instruction video](https://www.youtube.com/watch?v=neoK_WMq92g) (Assuming you have already installed uv)

Go to Claude > Settings > Developer > Edit Config > claude_desktop_config.json to include the following:

```json
{
    "mcpServers": {
        "blender": {
            "command": "uvx",
            "args": [
                "blender-mcp"
            ]
        }
    }
}
```

### Cursor integration

[![Install MCP Server](https://cursor.com/deeplink/mcp-install-dark.svg)](https://cursor.com/install-mcp?name=blender&config=eyJjb21tYW5kIjoidXZ4IGJsZW5kZXItbWNwIn0%3D)

For Mac users, go to Settings > MCP and paste the following 

- To use as a global server, use "add new global MCP server" button and paste
- To use as a project specific server, create `.cursor/mcp.json` in the root of the project and paste


```json
{
    "mcpServers": {
        "blender": {
            "command": "uvx",
            "args": [
                "blender-mcp"
            ]
        }
    }
}
```

For Windows users, go to Settings > MCP > Add Server, add a new server with the following settings:

```json
{
    "mcpServers": {
        "blender": {
            "command": "cmd",
            "args": [
                "/c",
                "uvx",
                "blender-mcp"
            ]
        }
    }
}
```

[Cursor setup video](https://www.youtube.com/watch?v=wgWsJshecac)

**⚠️ Only run one instance of the MCP server (either on Cursor or Claude Desktop), not both**

### Visual Studio Code Integration

_Prerequisites_: Make sure you have [Visual Studio Code](https://code.visualstudio.com/) installed before proceeding.

[![Install in VS Code](https://img.shields.io/badge/VS_Code-Install_blender--mcp_server-0098FF?style=flat-square&logo=visualstudiocode&logoColor=ffffff)](vscode:mcp/install?%7B%22name%22%3A%22blender-mcp%22%2C%22type%22%3A%22stdio%22%2C%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22blender-mcp%22%5D%7D)

### Installing the Blender Addon

1. Download the `addon.py` file from this repo
1. Open Blender
2. Go to Edit > Preferences > Add-ons
3. Click "Install..." and select the `addon.py` file
4. Enable the addon by checking the box next to "Interface: Blender MCP"


## Usage

### Starting the Connection
![BlenderMCP in the sidebar](assets/addon-instructions.png)

1. In Blender, go to the 3D View sidebar (press N if not visible)
2. Find the "BlenderMCP" tab
3. Turn on the Poly Haven checkbox if you want assets from their API (optional)
4. Click "Connect to Claude"
5. Make sure the MCP server is running in your terminal

### Using with Claude

Once the config file has been set on Claude, and the addon is running on Blender, you will see a hammer icon with tools for the Blender MCP.

![BlenderMCP in the sidebar](assets/hammer-icon.png)

### Vision Tools (vLLM + Chains)

- `register_vision_chain(name, chain_spec_json)`
  - Registers or updates a named chain. Example spec:
    ```json
    {
      "endpoint": "http://localhost:8000/v1/chat/completions",
      "models": {
        "type": "ring",
        "items": ["deepseek-ocr", "deepseek-ocr-vision"],
        "rotate_on_call": true
      },
      "prompt": "Analyze the image and return text.",
      "temperature": 0.0,
      "max_tokens": 512,
      "views": ["front","left","right","top","iso"],
      "filters": [
        {"type":"includes","value":"Invoice"},
        {"type":"regex","pattern":"\\\d{2,}/\\\d{2,}/\\\d{4}"},
        {"type":"scene:collision","object":"active","with":"any"},
        {"type":"scene:has_material","name_contains":"metal"}
      ]
    }
    ```

- `list_vision_chains()`
  - Lists registered chains with minimal info (endpoint/model(s)/views).

- `verify_vllm_connection(...)`
  - Pings the configured vLLM endpoint (or an override) using the new health-check
    flow. Logs the outcome to the MCP console and returns a JSON report.

- `vision_inspect_view(...)`
  - Params: `chain`, `chain_spec_json`, `models_csv`, `view`, `image_path`, `image_url`, `max_tokens`, `temperature`, `lighting`, `distance`, `filter_spec_json`, `prompt`
  - Behavior: captures a viewport shot (unless `image_*` provided), optionally positions camera/lighting, runs one or more models, and returns:
    - `scene_filters`: boolean scene checks in Blender
    - `results`: per-model `response` and `text_filters` outcomes

- `vision_multi_view(...)`
  - Params: `chain`, `chain_spec_json`, `views_csv`, `models_csv`, `map_reduce`, `lighting`, `distance`, `per_view_prompt`, `filter_spec_json`
  - Behavior: iterates over views, evaluates scene filters per view, runs models per view, and returns results with a simple `concat` aggregator.

Notes
- vLLM endpoint must be OpenAI-compatible `/v1/chat/completions`.
- Scene filters supported: `scene:collision` (AABB overlap heuristic) and `scene:has_material`.
- Text filters supported: `includes(value)`, `regex(pattern)`.

#### Configuration & Persistence

- Persistent settings now live at `~/.config/blender-mcp/settings.json` on Linux
  (or `%APPDATA%\blender-mcp\settings.json` on Windows). Override the path with
  `BLENDER_MCP_SETTINGS_PATH`.
- The `vllm.models` entry is stored as a **model ring** (`{"type":"ring", ...}`),
  enabling round-robin selection with optional rotation per call.
- Adjust `rotate_on_call` to cycle the primary model each invocation, or edit the
  `items` array to control the order.
- Run `python test_vllm_connection.py` (or the MCP tool `verify_vllm_connection`)
  to confirm the vLLM server is reachable; both routes log to the system console
  and exit non-zero if the endpoint cannot be contacted.

### Docker Compose Setup

- Use the provided `docker-compose.yml` to start both the MCP server and a vLLM
  OpenAI-compatible endpoint:
  ```bash
  docker compose up --build
  ```
- Override defaults by exporting:
  - `VLLM_MODEL_DIR` – host directory containing your model weights (mounted read-only at `/models`)
  - `VLLM_MODEL` – model identifier passed to `vllm.entrypoints.openai.api_server`
  - `VLLM_MAX_MODEL_LEN` – optional context length override
- The compose file maps Blender’s socket target to `host.docker.internal`; ensure
  the Blender addon listens on `9876` (default) and that the host mapping is
  supported on your platform.

#### Local vision model cache

The vLLM container no longer downloads checkpoints from Hugging Face at runtime.
Instead, pull models onto the host once and mount them into the container:

```bash
# install the Hugging Face CLI (once)
UV_CACHE_DIR="$HOME/.cache/uv" UV_TOOL_DIR="$HOME/.local/uv-tools" \
  uv tool install huggingface_hub[cli]

# download a model to ~/.models (change the repo ID if you prefer another build)
just hf-download model=liuhaotian/llava-v1.5-7b-hf dest=$HOME/.models/llava-v1.5-7b-hf

# start vLLM with the cached weights
just vllm-up

# follow logs / health status
just vllm-logs
```

By default `direnv` exports `VLLM_MODEL_DIR=$HOME/.models/llava-hf-llava-1.5-7b-hf`
and the compose file mounts that directory at `/models/llava`. Update
`VLLM_MODEL_DIR` and `VLLM_DTYPE` in `.envrc` if you want to point at a different
checkpoint or switch precision (for example `VLLM_DTYPE=float32` for CPU runs).

> **Tip**: The server still requires a compatible GPU for most checkpoints. If
> you switch to a CPU-only model, update `docker-compose.yml` accordingly.

#### Capabilities

- Get scene and object information 
- Create, delete and modify shapes
- Apply or create materials for objects
- Execute any Python code in Blender
- Download the right models, assets and HDRIs through [Poly Haven](https://polyhaven.com/)
- AI generated 3D models through [Hyper3D Rodin](https://hyper3d.ai/)


### Example Commands

Here are some examples of what you can ask Claude to do:

- "Create a low poly scene in a dungeon, with a dragon guarding a pot of gold" [Demo](https://www.youtube.com/watch?v=DqgKuLYUv00)
- "Create a beach vibe using HDRIs, textures, and models like rocks and vegetation from Poly Haven" [Demo](https://www.youtube.com/watch?v=I29rn92gkC4)
- Give a reference image, and create a Blender scene out of it [Demo](https://www.youtube.com/watch?v=FDRb03XPiRo)
- "Generate a 3D model of a garden gnome through Hyper3D"
- "Get information about the current scene, and make a threejs sketch from it" [Demo](https://www.youtube.com/watch?v=jxbNI5L7AH8)
- "Make this car red and metallic" 
- "Create a sphere and place it above the cube"
- "Make the lighting like a studio"
- "Point the camera at the scene, and make it isometric"

## Hyper3D integration

Hyper3D's free trial key allows you to generate a limited number of models per day. If the daily limit is reached, you can wait for the next day's reset or obtain your own key from hyper3d.ai and fal.ai.

## Troubleshooting

- **Connection issues**: Make sure the Blender addon server is running, and the MCP server is configured on Claude, DO NOT run the uvx command in the terminal. Sometimes, the first command won't go through but after that it starts working.
- **Timeout errors**: Try simplifying your requests or breaking them into smaller steps
- **Poly Haven integration**: Claude is sometimes erratic with its behaviour
- **Have you tried turning it off and on again?**: If you're still having connection errors, try restarting both Claude and the Blender server


## Technical Details

### Communication Protocol

The system uses a simple JSON-based protocol over TCP sockets:

- **Commands** are sent as JSON objects with a `type` and optional `params`
- **Responses** are JSON objects with a `status` and `result` or `message`

## Limitations & Security Considerations

- The `execute_blender_code` tool allows running arbitrary Python code in Blender, which can be powerful but potentially dangerous. Use with caution in production environments. ALWAYS save your work before using it.
- Poly Haven requires downloading models, textures, and HDRI images. If you do not want to use it, please turn it off in the checkbox in Blender. 
- Complex operations might need to be broken down into smaller steps


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This is a third-party integration and not made by Blender. Made by [Siddharth](https://x.com/sidahuj)
