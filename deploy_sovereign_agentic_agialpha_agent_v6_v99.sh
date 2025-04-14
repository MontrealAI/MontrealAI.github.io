#!/usr/bin/env bash
# deploy_sovereign_agentic_agialpha_agent_v6_v16.sh
#
# A single script that:
#   1) Creates a project folder (sovereign_agent_deploy/ by default).
#   2) Generates all required agent files (sovereign_agent.py, Dockerfile, etc.).
#   3) Optionally copies data directories for persistent storage (weaviate_data, chroma_storage, data).
#   4) Runs docker-compose build + docker-compose up -d
#   5) Automatically attaches to the "sovereign_agent" container in a user-friendly Terminal way.
#
# For high-stakes, production-ready deployment. Non-technical friendly.
#
# This version:
# - Installs cryptography (>=39.0.0) to fix the "cryptography library is required for signature verification" error.
# - Pins sentence-transformers / torch / transformers to avoid "init_empty_weights not defined" error.
# - Pins numpy<2.0.0.
# - Adds minimal Flask-based web chat at port 5000, with a token-gated webpage that requires 500k tokens for page access
#   and 1M tokens for agent access.
# - Ensures the user's front-end fetch calls the same origin ("/chat") instead of hardcoding "http://localhost:5000/chat".
# - Explicitly upgrades pip/setuptools/wheel inside the Docker image to fix “invalid command 'bdist_wheel'” issues.
# - Integrates Agentic Tree Search (`agentic_treesearch.py`) as a core planning engine for open-ended alpha exploration,
#   exposed at route `/tree_search` in the strictly web-based interface.
# - Returns BFS details (final expansions, best node, etc.) so the UI can display them.
# - **Extra diagnostics** in `/tree_search` route for debugging the HTTP 500 error.

set -e

TARGET_DIR="sovereign_agent_deploy"
OLLAMA_CONFIG_FILE="ollama_model.conf"  # to persist user’s model choice

###############################################################################
# 0) Check Port 11434 Freed (for Ollama)
###############################################################################
function check_port_11434_free() {
  if command -v lsof &> /dev/null; then
    PORT_CHECK="$(lsof -i tcp:11434 -sTCP:LISTEN -t 2>/dev/null || true)"
  else
    PORT_CHECK="$(netstat -anp 2>/dev/null | grep 11434 | grep LISTEN || true)"
  fi
  if [ -n "$PORT_CHECK" ]; then
    echo "Error: Port 11434 is already in use by another process. Please free this port and re-run."
    exit 1
  fi
}

###############################################################################
# 1) Start Ollama Daemon in background
###############################################################################
function start_ollama_daemon() {
  echo "Starting Ollama server in background..."
  nohup ollama serve > /tmp/ollama_server.log 2>&1 &
  OLLAMA_PID=$!

  READY=false
  for i in {1..15}; do
    sleep 1
    if command -v nc &> /dev/null; then
      if nc -z localhost 11434 2>/dev/null; then
        READY=true
        break
      fi
    else
      if curl -s http://localhost:11434/ > /dev/null; then
        READY=true
        break
      fi
    fi
  done

  if [ "$READY" != "true" ]; then
    echo "Error: Ollama server not responding on port 11434."
    kill $OLLAMA_PID 2>/dev/null || true
    exit 1
  fi
  echo "Ollama daemon is running on port 11434."
}

###############################################################################
# 2) Attempt to Upgrade Ollama if needed
###############################################################################
function upgrade_ollama() {
  unameOut="$(uname -s)"
  if [[ "$unameOut" == "Linux" ]]; then
    echo "Upgrading Ollama for Linux..."
    if ! command -v curl &> /dev/null; then
      echo "Error: 'curl' not installed. Please install it and re-run."
      exit 1
    fi
    if ! curl -fsSL https://ollama.com/install.sh | sh; then
      echo "Error: Failed to upgrade Ollama on Linux."
      exit 1
    fi
  elif [[ "$unameOut" == "Darwin" ]]; then
    echo "Upgrading Ollama on macOS..."
    if ! command -v brew &> /dev/null; then
      echo "Error: 'brew' not found. Please install Homebrew or see https://brew.sh/"
      exit 1
    fi
    if ! brew upgrade ollama; then
      echo "Error: Failed to upgrade Ollama via Homebrew."
      exit 1
    fi
  else
    echo "Unsupported OS for auto-upgrade. See https://ollama.com/download"
    exit 1
  fi
}

###############################################################################
# 3) Pull gemma3:4b if user wants that default
###############################################################################
function pull_gemma3() {
  echo "Pulling 'gemma3:4b' model (this may take several minutes)..."
  set +e
  PULL_OUTPUT="$(ollama pull gemma3:4b 2>&1)"
  PULL_STATUS=$?
  set -e

  if [ $PULL_STATUS -ne 0 ]; then
    echo "$PULL_OUTPUT"
    if echo "$PULL_OUTPUT" | grep -q "412:"; then
      echo "Detected version mismatch (412). Attempting Ollama upgrade..."
      upgrade_ollama
      killall ollama 2>/dev/null || true
      sleep 1
      check_port_11434_free
      start_ollama_daemon
      echo "Retrying pull of 'gemma3:4b'..."
      if ! ollama pull gemma3:4b; then
        echo "Error: second attempt to pull gemma3:4b also failed."
        killall ollama 2>/dev/null || true
        exit 1
      fi
    else
      echo "Error: failed to pull gemma3:4b"
      killall ollama 2>/dev/null || true
      exit 1
    fi
  fi
  echo "Successfully pulled 'gemma3:4b' model."
}

###############################################################################
# 4) Show Download Progress
###############################################################################
function show_download_progress() {
  while IFS= read -r line; do
    echo "$line"
  done
}

###############################################################################
# 5) Interactive Model Selection if no API keys
###############################################################################
function interactive_ollama_model_selection() {
  if [ -f "$OLLAMA_CONFIG_FILE" ]; then
    source "$OLLAMA_CONFIG_FILE" || true
    if [ -n "$OLLAMA_MODEL" ]; then
      echo "Using previously selected local Ollama model: \"$OLLAMA_MODEL\" (from $OLLAMA_CONFIG_FILE)."
      return
    fi
  fi

  echo "Fetching up to 20 popular models from Ollama library (ollama.com/library)..."
  local model_page
  model_page="$(curl -sSfL "https://ollama.com/library" || true)"
  if [ -z "$model_page" ]; then
    echo "⚠️  Unable to retrieve model list from ollama.com. Possibly no internet connection."
    echo -n "Enter the Ollama model name to use (e.g. llama2:7b or gemma3:4b). Press ENTER for 'gemma3:4b': "
    read -r user_input_model
    if [ -z "$user_input_model" ]; then
      user_input_model="gemma3:4b"
    fi
    OLLAMA_MODEL="$user_input_model"
  else
    local model_names
    model_names=$(echo "$model_page" | sed -n 's/.*href="\/library\/\([^"]*\)".*/\1/p' | head -n 20)
    if [ -z "$model_names" ]; then
      echo "No models found on the library page. Defaulting to gemma3:4b."
      OLLAMA_MODEL="gemma3:4b"
    else
      echo
      echo "Top 20 Ollama models (by popularity) found on the library page:"
      local i=1
      while IFS= read -r model; do
        local anchor_block
        anchor_block=$(echo "$model_page" | grep -A10 -m1 "href=\"/library/$model\"" | sed ':a;N;$!ba;s/\n/ /g')
        local anchor_block_sans_tags
        anchor_block_sans_tags=$(echo "$anchor_block" | sed -E 's/<[^>]+>/ /g' | tr -s ' ')
        local size_info
        size_info=$(echo "$anchor_block_sans_tags" | grep -oE '[0-9]+(\.[0-9]+)?[MG]B' | head -n1)
        [ -n "$size_info" ] && size_info="(Size: $size_info)" || size_info="(Size: unknown)"
        echo "  $i) $model $size_info"
        i=$((i+1))
      done <<< "$model_names"

      echo
      echo -n "Enter the number of the model to use, or type a custom name. Press ENTER for 'gemma3:4b': "
      read -r choice
      if [[ -z "$choice" ]]; then
        OLLAMA_MODEL="gemma3:4b"
      elif [[ "$choice" =~ ^[0-9]+$ ]]; then
        local total_count
        total_count=$(echo "$model_names" | wc -l)
        if (( choice >= 1 && choice <= total_count )); then
          OLLAMA_MODEL=$(echo "$model_names" | sed -n "${choice}p")
          echo "Selected model #$choice: \"$OLLAMA_MODEL\"."
        else
          echo "Invalid selection ($choice). Defaulting to 'gemma3:4b'."
          OLLAMA_MODEL="gemma3:4b"
        fi
      else
        OLLAMA_MODEL="$choice"
      fi
    fi
  fi

  echo "Using Ollama model: \"$OLLAMA_MODEL\""
  echo "export OLLAMA_MODEL=\"$OLLAMA_MODEL\"" > "$OLLAMA_CONFIG_FILE"
  echo "(Saved choice to $OLLAMA_CONFIG_FILE for future runs.)"
}

###############################################################################
# MAIN SCRIPT
###############################################################################
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "No OPENAI_API_KEY or ANTHROPIC_API_KEY found. Using local Ollama (interactive selection)..."

  if command -v lsof &> /dev/null; then
    sudo kill -9 "$(sudo lsof -t -i:11434)" 2>/dev/null || true
  fi

  if ! command -v ollama &> /dev/null; then
    echo "Ollama not found on the host system, installing system-wide..."
    unameOut="$(uname -s)"
    case "$unameOut" in
      Linux*)
        echo "Installing Ollama for Linux..."
        if ! command -v curl &> /dev/null; then
          echo "Error: 'curl' not installed. Please install it and re-run."
          exit 1
        fi
        if ! curl -fsSL https://ollama.com/install.sh | sh; then
          echo "Error: Ollama install failed on Linux."
          exit 1
        fi
        ;;
      Darwin*)
        echo "Installing Ollama for macOS via Homebrew..."
        if ! command -v brew &> /dev/null; then
          echo "Error: 'brew' not found. Please install Homebrew or see https://brew.sh/"
          exit 1
        fi
        if ! brew install ollama; then
          echo "Error: brew install ollama failed."
          exit 1
        fi
        ;;
      *)
        echo "Unsupported OS. See https://github.com/jmorganca/ollama for instructions."
        exit 1
        ;;
    esac
  else
    echo "Ollama is already installed on the host."
  fi

  check_port_11434_free
  start_ollama_daemon
  interactive_ollama_model_selection

  echo "Exporting LLM_PROVIDER=ollama, OLLAMA_MODEL=${OLLAMA_MODEL}"
  export LLM_PROVIDER="ollama"
  export OLLAMA_MODEL="${OLLAMA_MODEL}"
  export OLLAMA_URL="http://localhost:11434"

  echo "Pulling Ollama model \"$OLLAMA_MODEL\". This may take a while if not already downloaded..."
  set +e
  PULL_OUT="$(ollama pull "$OLLAMA_MODEL" | show_download_progress)"
  PULL_OK=${PIPESTATUS[0]}
  set -e

  if [ $PULL_OK -ne 0 ]; then
    if echo "$PULL_OUT" | grep -q "412:"; then
      echo "Detected version mismatch. Attempting Ollama upgrade..."
      upgrade_ollama
      killall ollama 2>/dev/null || true
      sleep 1
      check_port_11434_free
      start_ollama_daemon
      echo "Retrying pull of '$OLLAMA_MODEL'..."
      if ! ollama pull "$OLLAMA_MODEL"; then
        echo "Error: second attempt to pull $OLLAMA_MODEL also failed."
        killall ollama 2>/dev/null || true
        exit 1
      fi
    else
      echo "Error: failed to pull $OLLAMA_MODEL"
      killall ollama 2>/dev/null || true
      exit 1
    fi
  fi
else
  echo "Found OPENAI_API_KEY or ANTHROPIC_API_KEY; using cloud-based LLM provider."
fi

echo "============================================================"
echo "Sovereign Agentic AGI ALPHA - Automatic Deployment Script"
echo "============================================================"
echo "Creating '$TARGET_DIR/', generating agent files, and"
echo "running docker-compose build/up -d. Make sure Docker &"
echo "Compose are installed, and Python 3.9+ for local usage."
echo "------------------------------------------------------------"
echo

mkdir -p "$TARGET_DIR"
echo "[1/5] Created/verified directory: '$TARGET_DIR'."

echo "[2/5] Generating files..."

###############################################################################
# The main agent code
###############################################################################
cat <<'AGENT_PY' > "$TARGET_DIR/sovereign_agent.py"
#!/usr/bin/env python3
"""
Sovereign Agentic AGI ALPHA - Single-File Monolithic Version
(with chain-of-thought, advanced reasoning, MCP extension,
 pinned sentence-transformers + cryptography, etc.)

We do NOT remove or alter existing Terminal-based chat.
We ADD a minimal Flask-based web interface that:
  1) Serves a "/" route with the big HTML page
  2) Serves a "/chat" endpoint for JSON POST, so user can chat from the webpage
  3) Demonstrates a "/tree_search" endpoint for open-ended alpha exploration
     via the integrated "Agentic Tree Search".

**Extra Diagnostics**: The "/tree_search" route logs directory listings, writes a
small JSON config, and logs the full traceback if there's an error, to help
debug 500 errors.

**Additionally**: We now return BFS details (expanded nodes, best node, etc.)
so the front-end can display them directly (via `resultDetails`).
"""

import os
import sys
import json
import yaml
import requests
import time
import logging
import subprocess
import asyncio
import socket
import uuid
import threading
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

logger = logging.getLogger("SovereignAlphaAgent")
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Attempt optional openai
try:
    import openai
except ImportError:
    openai = None

# Attempt optional anthropic
try:
    from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
except ImportError:
    Anthropic = None
    HUMAN_PROMPT = None
    AI_PROMPT = None

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None
    ChromaSettings = None

try:
    import weaviate
except ImportError:
    weaviate = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import dns.resolver
except ImportError:
    dns = None

try:
    from zeroconf import Zeroconf, ServiceBrowser, ServiceStateChange, ServiceInfo
except ImportError:
    Zeroconf = None
    ServiceBrowser = None
    ServiceInfo = None

try:
    import cryptography
except ImportError:
    cryptography = None

# We'll import Flask for the minimal web chat + endpoint for tree search:
try:
    from flask import Flask, request, jsonify
except ImportError:
    Flask = None

# Import the agentic treesearch module (we place it in the same directory).
try:
    import agentic_treesearch
except ImportError:
    agentic_treesearch = None

trust_file = "trust_whitelist.yaml"
trust_config = {
    "allowed_domains": [],
    "allowed_paths": [],
    "allowed_tools": [],
    "allowed_cert_fingerprints": [],
    "allowed_ed25519_keys": []
}
if os.path.isfile(trust_file):
    try:
        with open(trust_file, "r") as f:
            user_trust = yaml.safe_load(f)
        for k in trust_config.keys():
            if k in user_trust and isinstance(user_trust[k], list):
                trust_config[k] = user_trust[k]
    except Exception as e:
        logger.error(f"Error reading trust_whitelist.yaml: {e}")
else:
    with open(trust_file, "w") as f:
        yaml.safe_dump(trust_config, f)
    logger.info(f"Created default {trust_file}.")

def is_domain_allowed(domain: str) -> bool:
    domain = domain.lower().strip()
    for allowed in trust_config["allowed_domains"]:
        if domain == allowed or domain.endswith('.' + allowed):
            return True
    return False

def is_path_allowed(path: str) -> bool:
    abspath = os.path.abspath(path)
    for allowed in trust_config["allowed_paths"]:
        allowed_abs = os.path.abspath(allowed)
        if abspath == allowed_abs or (
            abspath.startswith(allowed_abs) and
            (len(abspath) == len(allowed_abs) or abspath[len(allowed_abs)] == os.sep)
        ):
            return True
    return False

def plugin_web_fetch(url: str):
    parsed = urlparse(url)
    domain = parsed.netloc or parsed.path
    if not is_domain_allowed(domain):
        return f"Error: domain '{domain}' not allowed."
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return f"HTTP error {r.status_code}"
        text = r.text
        if len(text) > 1000:
            text = text[:1000] + "... [truncated]"
        return text
    except Exception as e:
        return f"Error: {e}"

def plugin_file_read(path: str):
    if not is_path_allowed(path):
        return f"Error: file path '{path}' not allowed."
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
            if len(data) > 1000:
                data = data[:1000] + "... [truncated]"
            return data
    except Exception as e:
        return f"Error: {e}"

def plugin_file_write(path: str, content: str):
    if not is_path_allowed(path):
        return f"Error: file path '{path}' not allowed."
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {path}."
    except Exception as e:
        return f"Error: {e}"

allowed_tools = trust_config.get("allowed_tools", [])
local_plugins = [
    {"name": "web_fetch", "description": "Fetch URL content", "func": plugin_web_fetch,
     "parameters": {"url": "string"}},
    {"name": "file_read", "description": "Read text from a file", "func": plugin_file_read,
     "parameters": {"path": "string"}},
    {"name": "file_write","description": "Write text to a file","func": plugin_file_write,
     "parameters": {"path": "string", "content": "string"}}
]
local_plugins = [p for p in local_plugins if p["name"] in allowed_tools]

def create_openai_functions(plugin_list):
    out = []
    for p in plugin_list:
        props = {}
        required = []
        for arg, typ in p["parameters"].items():
            props[arg] = {"type": "string"}
            required.append(arg)
        out.append({
            "name": p["name"],
            "description": p["description"],
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required
            }
        })
    return out

openai_functions = create_openai_functions(local_plugins)

class VectorMemory:
    def __init__(self):
        self.backend = os.getenv("MEMORY_BACKEND","chroma").lower()
        self._embed_func = None
        embed_mode = os.getenv("EMBEDDING_MODE","").lower()

        use_openai_embed = False
        try:
            if embed_mode == "openai" and openai and os.getenv("OPENAI_API_KEY"):
                use_openai_embed = True
            elif embed_mode == "local":
                use_openai_embed = False
            else:
                if openai and os.getenv("OPENAI_API_KEY"):
                    use_openai_embed = True
        except:
            pass

        if use_openai_embed and openai:
            def openai_embed(text: str):
                try:
                    resp = openai.Embedding.create(model="text-embedding-ada-002", input=[text])
                    return resp["data"][0]["embedding"]
                except Exception as e:
                    logger.error(f"OpenAI embedding error: {e}")
                    return None
            self._embed_func = openai_embed
        else:
            if SentenceTransformer:
                try:
                    modelname = os.getenv("EMBEDDING_MODEL","sentence-transformers/all-MiniLM-L6-v2")
                    logger.info(f"Loading local embedding model {modelname}...")
                    st = SentenceTransformer(modelname)
                    self._embed_func = lambda txt: st.encode(txt).tolist()
                except Exception as e:
                    logger.error(f"Local embedding init failed: {e}")
                    self._embed_func = lambda txt: [float(hash(txt)%100000)]
            else:
                logger.warning("No SentenceTransformer installed; using dummy embeddings.")
                self._embed_func = lambda txt: [float(hash(txt)%100000)]

        self.chroma_client = None
        self.collection = None
        self.weaviate_client = None

        if self.backend == "chroma" and chromadb:
            dir_ = os.getenv("CHROMA_PERSIST_DIR","chroma_storage")
            try:
                cset = ChromaSettings(persist_directory=dir_)
                self.chroma_client = chromadb.Client(cset)
                self.collection = self.chroma_client.get_or_create_collection(name="sovereign_memory")
            except Exception as e:
                logger.error(f"Chroma init failed: {e}")
                self.chroma_client = None
        elif self.backend == "weaviate" and weaviate:
            url = os.getenv("WEAVIATE_URL","http://localhost:8080")
            auth = None
            if os.getenv("WEAVIATE_API_KEY"):
                import weaviate.auth
                auth = weaviate.auth.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY"))
            try:
                self.weaviate_client = weaviate.Client(url, auth_client_secret=auth)
                schema = self.weaviate_client.schema.get()
                classes = [c["class"] for c in schema.get("classes",[])]
                if "MemoryItem" not in classes:
                    class_schema = {
                        "class":"MemoryItem",
                        "vectorizer":"none",
                        "properties":[{"name":"content","dataType":["text"]}]
                    }
                    self.weaviate_client.schema.create_class(class_schema)
            except Exception as e:
                logger.error(f"Weaviate init failed: {e}")
                self.weaviate_client = None

    def add(self, text: str):
        if not text or not self._embed_func:
            return
        vec = self._embed_func(text)
        if vec is None:
            return
        if self.backend=="chroma" and self.chroma_client:
            try:
                self.collection.add(documents=[text], embeddings=[vec], ids=[str(uuid.uuid4())])
            except Exception as e:
                logger.error(f"Chroma add error: {e}")
        elif self.backend=="weaviate" and self.weaviate_client:
            try:
                self.weaviate_client.data_object.create({"content": text}, "MemoryItem", vector=vec)
            except Exception as e:
                logger.error(f"Weaviate add error: {e}")

    def search(self, query: str, k=3) -> List[str]:
        if not query or not self._embed_func:
            return []
        vec = self._embed_func(query)
        if vec is None:
            return []
        if self.backend=="chroma" and self.chroma_client:
            try:
                res = self.collection.query(query_embeddings=[vec], n_results=k, include=["documents"])
                if res and "documents" in res and res["documents"]:
                    return res["documents"][0]
            except Exception as e:
                logger.error(f"Chroma search error: {e}")
        elif self.backend=="weaviate" and self.weaviate_client:
            try:
                r = self.weaviate_client.query.get("MemoryItem", ["content"]) \
                           .with_near_vector({"vector":vec}).with_limit(k).do()
                items = r.get("data",{}).get("Get",{}).get("MemoryItem",[])
                return [it["content"] for it in items if "content" in it]
            except Exception as e:
                logger.error(f"Weaviate search error: {e}")
        return []

import re

class ReasoningAgent:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER","ollama").lower()
        self.model_name = (os.getenv("OLLAMA_MODEL","gemma3:4b")
                           if self.provider == "ollama"
                           else os.getenv("OPENAI_MODEL","gpt-3.5-turbo"))
        self.history = []
        self.memory = VectorMemory()
        self.debug_mode = False

        self.anthropic_client = None
        if self.provider == "anthropic" and Anthropic is not None and os.getenv("ANTHROPIC_API_KEY"):
            self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        if self.provider == "openai" and openai is not None and os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")

        self.max_steps = 5
        self.reaction_tools = {
            "Search": self.tool_search,
            "Calculate": self.tool_calculate
        }

    def tool_search(self, query: str) -> str:
        try:
            r = requests.get(f"https://api.duckduckgo.com/?q={query}&format=json").json()
            abstract = r.get("Abstract", "")
            related = r.get("RelatedTopics", [])
            if abstract:
                return abstract[:300]
            elif related:
                return related[0].get("Text","")[:300]
            else:
                return "No relevant info."
        except Exception as e:
            return f"[Search error: {e}]"

    def tool_calculate(self, expression: str) -> str:
        try:
            return str(eval(expression, {"__builtins__": {}}, {}))
        except Exception as e:
            return f"[Calculation error: {e}]"

    def generate_prompt(self, user_input: str) -> str:
        system_text = (
            "You are the Sovereign Agentic AGI ALPHA. "
            "You think step-by-step but never reveal your chain-of-thought. "
            "Use the following format for advanced reasoning if needed:\n"
            "Thought: <reasoning>\n"
            "Action: <tool> [<input>]\n"
            "Observation: <result>\n"
            "Thought: <further reasoning>\n"
            "Answer: <final answer>\n\n"
            "Tools available: Search, Calculate.\n"
        )
        conv_text = ""
        for turn in self.history:
            if turn["role"] == "user":
                conv_text += f"User: {turn['content']}\n"
            else:
                conv_text += f"Assistant: {turn['content']}\n"
        conv_text += f"User: {user_input}\nAssistant:"
        return system_text + conv_text

    def call_llm(self, prompt: str) -> str:
        if self.provider == "openai":
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[{"role":"system","content":prompt}],
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"[OpenAI error: {e}]"
        elif self.provider == "anthropic" and self.anthropic_client:
            full_prompt = f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}"
            try:
                resp = self.anthropic_client.completions.create(
                    model=os.getenv("ANTHROPIC_MODEL","claude-2"),
                    prompt=full_prompt,
                    max_tokens_to_sample=1024,
                    temperature=0.7
                )
                return resp.completion.strip()
            except Exception as e:
                return f"[Anthropic error: {e}]"
        else:
            # local Ollama
            cmd = ["ollama", "run", self.model_name, prompt]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    return result.stdout.strip()
                else:
                    err = result.stderr.strip() or "Unknown ollama CLI error"
                    return f"[Ollama error: {err}]"
            except FileNotFoundError:
                # fallback to HTTP
                try:
                    url = os.getenv("OLLAMA_URL","http://localhost:11434") + "/generate"
                    pl = {"model": self.model_name, "prompt": prompt}
                    r = requests.post(url, json=pl, timeout=300)
                    if r.status_code == 200:
                        j = r.json()
                        return j.get("reply", j.get("completion","")).strip()
                    else:
                        return f"[Ollama HTTP error: {r.status_code}]"
                except Exception as e:
                    return f"[Ollama fallback error: {e}]"
            except Exception as e:
                return f"[Ollama error: {e}]"

    def parse_react(self, text: str):
        lines = text.split("\n")
        steps = []
        current_block = {"type": None, "content": ""}
        for line in lines:
            line = line.strip()
            if line.lower().startswith("thought:"):
                if current_block["type"] is not None:
                    steps.append(current_block)
                current_block = {"type": "thought", "content": line[len("Thought:"):].strip()}
            elif line.lower().startswith("action:"):
                if current_block["type"] is not None:
                    steps.append(current_block)
                current_block = {"type": "action", "content": line[len("Action:"):].strip()}
            elif line.lower().startswith("observation:"):
                if current_block["type"] is not None:
                    steps.append(current_block)
                current_block = {"type": "observation", "content": line[len("Observation:"):].strip()}
            elif line.lower().startswith("answer:"):
                if current_block["type"] is not None:
                    steps.append(current_block)
                current_block = {"type": "answer", "content": line[len("answer:"):].strip()}
                steps.append(current_block)
                current_block = {"type": None, "content": ""}
            else:
                if current_block["type"] is not None:
                    current_block["content"] += " " + line
        if current_block["type"] is not None:
            steps.append(current_block)
        return steps

    def run_react_loop(self, prompt: str) -> str:
        remaining_steps = self.max_steps
        partial_prompt = prompt
        while remaining_steps > 0:
            model_output = self.call_llm(partial_prompt)
            steps = self.parse_react(model_output)
            final_answer = ""
            tool_invoked = False
            for step in steps:
                if step["type"] == "answer":
                    final_answer = step["content"]
                    break
                elif step["type"] == "action":
                    toolline = step["content"]
                    match = re.match(r"(\w+)\s*\[(.*)\]", toolline)
                    if match:
                        toolname = match.group(1).strip()
                        toolarg = match.group(2).strip()
                        toolfn = self.reaction_tools.get(toolname, None)
                        if toolfn:
                            observation = toolfn(toolarg)
                            partial_prompt += f"\nObservation: {observation}\nThought:"
                            tool_invoked = True
                        else:
                            partial_prompt += f"\nObservation: [No such tool: {toolname}]\nThought:"
                            tool_invoked = True
            if final_answer:
                return final_answer
            if not tool_invoked:
                # no further "Action" => agent likely gave a final answer
                return model_output.strip()
            remaining_steps -= 1
        return "I'm sorry, I cannot determine the final answer."

    def handle_message(self, user_input: str) -> str:
        self.memory.add(user_input)
        prompt = self.generate_prompt(user_input)
        final_answer = self.run_react_loop(prompt)
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": final_answer})
        return final_answer

def start_flask_server(agent):
    """
    Minimal Flask-based server on port 5000 so the user
    can chat from the webpage with the existing agent.
    Also serve a GET "/" route with a big HTML page (sovereign_agent_web.html).
    Also demonstrate a "/tree_search" endpoint for Agentic Tree Search debugging.
    Returns BFS expansions and best path so the UI can display them.
    """
    if not Flask:
        logger.warning("Flask not installed. Web chat is disabled.")
        return

    app = Flask("SovereignAgentFlask")

    PAGE_FILE = os.path.join(os.path.dirname(__file__), "sovereign_agent_web.html")
    if not os.path.isfile(PAGE_FILE):
        logger.warning("No 'sovereign_agent_web.html' found, web interface won't work.")
        PAGE_CONTENT = "<h1>Missing sovereign_agent_web.html</h1>"
    else:
        with open(PAGE_FILE, "r", encoding="utf-8") as f:
            PAGE_CONTENT = f.read()

    @app.route("/", methods=["GET"])
    def index():
        return PAGE_CONTENT, 200

    @app.route("/chat", methods=["POST"])
    def chat():
        data = request.json or {}
        user_input = data.get("message","")
        if not user_input:
            return jsonify({"reply": "No message received."}), 400
        reply = agent.handle_message(user_input)
        return jsonify({"reply": reply})

    @app.route("/tree_search", methods=["POST"])
    def tree_search():
        """
        A minimal demonstration endpoint that calls agentic_treesearch for advanced alpha exploration.
        We'll parse a "topic" from JSON, create a dummy JSON config,
        run agentic_treesearch.perform_experiments_bfts(config_path),
        log the BFS details, and return them so the front-end can display expansions & best path.
        """
        if not agentic_treesearch:
            return jsonify({"error": "Agentic Tree Search not installed."}), 500
        data = request.json or {}
        topic = data.get("topic","Test")
        try:
            logger.info("Ensuring 'workspace' directory exists...")
            os.makedirs("workspace", exist_ok=True)
            logger.info("Listing 'workspace' contents before tree search:")
            for f in os.listdir("workspace"):
                logger.info(f"   -> {f}")

            config_path = f"./dummy_config_for_{topic}.json"
            dummy_data = {"topic": topic, "ts": time.time()}
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(dummy_data, f)
            logger.info(f"Wrote dummy config {config_path}: {dummy_data}")

            logger.info(f"Calling agentic_treesearch.perform_experiments_bfts({config_path})...")
            bfs_details = agentic_treesearch.perform_experiments_bfts(config_path)
            logger.info("Tree search completed.")

            # Return BFS details so the front-end can show expansions & best path
            return jsonify({
                "result": f"Agentic Tree Search triggered with topic='{topic}'.",
                "resultDetails": bfs_details
            })
        except Exception as e:
            logger.exception("Error in /tree_search route", exc_info=e)
            return jsonify({"error": str(e)}), 500

    def run_flask():
        app.run(host="0.0.0.0", port=5000, debug=False)

    threading.Thread(target=run_flask, daemon=True).start()

def main():
    agent = ReasoningAgent()
    print("Sovereign Agentic AGI ALPHA with CoT + ReAct + MCP + Agentic Tree Search. Type 'exit' or 'quit' to quit.\n")

    # Attempt MCP extension
    try:
        import sovereign_alpha_mcp_extension as mcp_ext
        from sovereign_alpha_mcp_extension import TrustPolicy

        policy = TrustPolicy(
            allowed_cert_fingerprints=trust_config.get("allowed_cert_fingerprints", []),
            allowed_ed25519_keys=trust_config.get("allowed_ed25519_keys", []),
            allowed_domains=trust_config.get("allowed_domains", [])
        )
        extension = mcp_ext.SovereignAlphaAgentMCPExtension(
            trust_policy=policy,
            discovery_github=True,
            discovery_dnssd=True,
            discovery_referrals=[]
        )
        extension.discover_and_launch()
        logger.info(f"MCP Extension integrated. Found {len(extension.mcp_servers)} trusted MCP servers.")
    except Exception as ex:
        logger.warning(f"Could not initialize MCP Extension: {ex}")

    # Start minimal Flask server for web chat + static page + tree_search
    start_flask_server(agent)

    # Original Terminal-based chat loop
    while True:
        try:
            user_input = input("User> ")
        except EOFError:
            print("\n[EOF: stopping agent]")
            break
        if not user_input:
            continue
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Agent> Goodbye.")
            break
        response = agent.handle_message(user_input)
        print(f"Agent> {response}")

if __name__ == "__main__":
    main()
AGENT_PY

###############################################################################
# The Sovereign Alpha Agent MCP Extension Implementation
###############################################################################
cat <<'MCP_PY' > "$TARGET_DIR/sovereign_alpha_mcp_extension.py"
#!/usr/bin/env python3

import os
import json
import logging
import base64
import subprocess
import asyncio
import socket
from typing import List, Optional, Dict, Any

try:
    import dns
    import dns.resolver
except ImportError:
    dns = None

try:
    from zeroconf import Zeroconf, ServiceBrowser, ServiceStateChange, ServiceInfo
except ImportError:
    Zeroconf = None
    ServiceBrowser = None
    ServiceInfo = None

try:
    from cryptography.hazmat.primitives.serialization import load_pem_x509_certificate
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ed25519, rsa, ec, padding as asym_padding
    from cryptography.exceptions import InvalidSignature
except ImportError:
    raise RuntimeError("cryptography library is required for signature verification")

try:
    import requests
except ImportError:
    raise RuntimeError("requests library is required for HTTP operations in discovery")

logger = logging.getLogger("SovereignAlphaMCP")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

class TrustPolicy:
    """
    Trust policy for MCP plugins. Allows:
    - X.509 certificate fingerprint (SHA-256)
    - Ed25519 key (full or hashed)
    - Domain suffix
    """
    def __init__(
        self,
        allowed_cert_fingerprints: Optional[List[str]] = None,
        allowed_ed25519_keys: Optional[List[str]] = None,
        allowed_domains: Optional[List[str]] = None
    ):
        self.allowed_cert_fingerprints = [fp.lower().replace(':', '') for fp in (allowed_cert_fingerprints or [])]
        allowed_keys = allowed_ed25519_keys or []
        normalized_keys = []
        for key in allowed_keys:
            # If the key is a hex string 64 or 128 chars, treat it as raw hex.
            if all(c in "0123456789abcdefABCDEF" for c in key) and len(key) in (64, 128):
                normalized_keys.append(key.lower())
            else:
                # Attempt base64 decode. If fail, skip or log error.
                try:
                    key_bytes = base64.b64decode(key.strip())
                    normalized_keys.append(key_bytes.hex())
                except Exception as e:
                    logger.error(f"Invalid Ed25519 key format: {key} ({e})")
        self.allowed_ed25519_keys = normalized_keys
        # domain suffixes, e.g. "example.com" or "mydomain.org"
        self.allowed_domains = [d.lower().lstrip("*").lstrip(".") for d in (allowed_domains or [])]

    def is_trusted(self, plugin: Dict[str, Any]) -> bool:
        source = plugin.get("source","<unknown>")
        name = plugin.get("name", source)
        cert_pem = plugin.get("certificate")
        signature_b64 = plugin.get("signature")
        pubkey_b64 = plugin.get("public_key")

        domain = None
        if plugin.get("url"):
            try:
                from urllib.parse import urlparse
                parsed = urlparse(plugin["url"])
                if parsed.hostname:
                    domain = parsed.hostname.lower()
            except:
                domain = None
        if not domain and plugin.get("domain"):
            domain = plugin["domain"].lower()

        # Check domain allowlist
        if domain:
            for allowed in self.allowed_domains:
                if domain == allowed or domain.endswith("." + allowed):
                    logger.info(f"TrustPolicy: '{name}' trusted by domain: {domain}")
                    return True

        # Check certificate fingerprint
        if cert_pem:
            try:
                cert = load_pem_x509_certificate(cert_pem.encode('utf-8'))
            except Exception as e:
                logger.error(f"TrustPolicy: Failed to load cert for plugin '{name}': {e}")
                return False
            try:
                cert_fingerprint = cert.fingerprint(hashes.SHA256()).hex()
            except Exception:
                cert_bytes = cert.tbs_certificate_bytes
                digest = hashes.Hash(hashes.SHA256())
                digest.update(cert_bytes)
                cert_fingerprint = digest.finalize().hex()
            if cert_fingerprint.lower() in self.allowed_cert_fingerprints:
                if signature_b64:
                    data = self._signature_data(plugin)
                    if self._verify_signature_with_cert(cert, signature_b64, data):
                        logger.info(f"TrustPolicy: '{name}' verified via certificate signature.")
                        return True
                    else:
                        logger.warning(f"TrustPolicy: Cert signature failed for '{name}'.")
                        return False
                else:
                    logger.info(f"TrustPolicy: '{name}' trusted (cert fingerprint match, no sig).")
                    return True
            else:
                logger.warning(f"TrustPolicy: Cert fingerprint not in allowlist for '{name}'.")

        # Check Ed25519 public key
        if pubkey_b64:
            try:
                pubkey_bytes = base64.b64decode(pubkey_b64.strip())
                pubkey_hex = pubkey_bytes.hex()
            except Exception as e:
                logger.error(f"TrustPolicy: Invalid public_key encoding for '{name}': {e}")
                return False
            allowed_key = False
            if pubkey_hex in self.allowed_ed25519_keys:
                allowed_key = True
            else:
                digest = hashes.Hash(hashes.SHA256())
                digest.update(pubkey_bytes)
                key_hash_hex = digest.finalize().hex()
                if key_hash_hex in self.allowed_ed25519_keys:
                    allowed_key = True
            if allowed_key:
                if signature_b64:
                    data = self._signature_data(plugin)
                    if self._verify_signature_ed25519(pubkey_bytes, signature_b64, data):
                        logger.info(f"TrustPolicy: '{name}' verified via Ed25519 signature.")
                        return True
                    else:
                        logger.warning(f"TrustPolicy: Ed25519 signature fail for '{name}'.")
                        return False
                else:
                    logger.info(f"TrustPolicy: '{name}' trusted (Ed25519 key match, no sig).")
                    return True
            else:
                logger.warning(f"TrustPolicy: Ed25519 key not in allowlist for '{name}'.")

        logger.warning(f"TrustPolicy: '{name}' NOT trusted (deny by default).")
        return False

    def _signature_data(self, plugin: Dict[str, Any]) -> bytes:
        data_obj = {k: v for k,v in plugin.items() if k not in ("signature","certificate")}
        try:
            return json.dumps(data_obj, sort_keys=True, separators=(',',':')).encode('utf-8')
        except:
            safe_obj = {}
            for k,v in data_obj.items():
                if isinstance(v,(str,int,float,bool,list,dict)) or v is None:
                    safe_obj[k] = v
            return json.dumps(safe_obj, sort_keys=True, separators=(',',':')).encode('utf-8')

    def _verify_signature_ed25519(self, pubkey_bytes: bytes, signature_b64: str, data: bytes) -> bool:
        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.exceptions import InvalidSignature
        try:
            signature = base64.b64decode(signature_b64.strip())
            ed25519.Ed25519PublicKey.from_public_bytes(pubkey_bytes).verify(signature, data)
            return True
        except (InvalidSignature, Exception):
            return False

    def _verify_signature_with_cert(self, cert, signature_b64: str, data: bytes) -> bool:
        from cryptography.hazmat.primitives.asymmetric import ed25519, rsa, ec, padding as asym_padding
        from cryptography.hazmat.primitives import hashes
        from cryptography.exceptions import InvalidSignature
        try:
            signature = base64.b64decode(signature_b64.strip())
        except Exception as e:
            logger.error(f"TrustPolicy: Error decoding base64 sig: {e}")
            return False
        pubkey = cert.public_key()
        try:
            if isinstance(pubkey, ed25519.Ed25519PublicKey):
                pubkey.verify(signature, data)
                return True
            elif isinstance(pubkey, rsa.RSAPublicKey):
                pubkey.verify(signature, data, asym_padding.PKCS1v15(), hashes.SHA256())
                return True
            elif isinstance(pubkey, ec.EllipticCurvePublicKey):
                pubkey.verify(signature, data, ec.ECDSA(hashes.SHA256()))
                return True
            else:
                logger.error("TrustPolicy: Unsupported key type in cert.")
                return False
        except InvalidSignature:
            return False
        except Exception as e:
            logger.error(f"TrustPolicy: Cert sig verify error: {e}")
            return False

class SovereignAlphaAgentMCPExtension:
    """
    Discover, validate, launch MCP servers.
    """
    def __init__(self, trust_policy: TrustPolicy,
                 discovery_github: bool = True,
                 discovery_dnssd: bool = True,
                 discovery_referrals: Optional[List[str]] = None):
        self.trust_policy = trust_policy
        self.discovery_github = discovery_github
        self.discovery_dnssd = discovery_dnssd
        self.discovery_referrals = discovery_referrals or []
        self.mcp_servers: List[Any] = []
        self._launched_containers: List[str] = []
        self._zeroconf: Optional[Zeroconf] = None

    def discover_and_launch(self) -> None:
        plugins = []
        if self.discovery_github:
            plugins += self._discover_from_github()
        if self.discovery_dnssd:
            plugins += self._discover_from_dnssd()
        if self.discovery_referrals:
            plugins += self._process_referrals(self.discovery_referrals)
        if not plugins:
            logger.info("No MCP plugins discovered from any source.")

        for plugin in plugins:
            name = plugin.get("name") or plugin.get("source") or "<unknown>"
            try:
                if not self.trust_policy.is_trusted(plugin):
                    logger.info(f"Skipping '{name}' (not trusted).")
                    continue
                transport = plugin.get("transport", "").lower()
                if not transport:
                    if plugin.get("command"):
                        transport = "stdio"
                    elif plugin.get("url"):
                        transport = "sse"
                    elif plugin.get("image"):
                        transport = "docker"

                if transport == "stdio":
                    params: Dict[str,Any] = {}
                    for k in ("command","args","env","cwd","encoding","encoding_error_handler"):
                        if k in plugin:
                            params[k] = plugin[k]
                    if "command" not in params:
                        logger.error(f"Cannot launch '{name}': no 'command' for stdio.")
                        continue
                    logger.info(f"Launched STDIO server for '{name}' (Pseudo).")
                    self.mcp_servers.append({"server_type":"stdio","params":params,"name":name})

                elif transport == "sse":
                    params: Dict[str,Any] = {}
                    for k in ("url","headers","timeout","sse_read_timeout"):
                        if k in plugin:
                            params[k] = plugin[k]
                    if "url" not in params:
                        logger.error(f"Cannot connect '{name}': no 'url' for SSE.")
                        continue
                    logger.info(f"Connected SSE server '{name}' at {params.get('url')}. (Pseudo)")
                    self.mcp_servers.append({"server_type":"sse","params":params,"name":name})

                elif transport == "docker":
                    image = plugin.get("image")
                    if not image:
                        logger.error(f"Cannot launch '{name}': no 'image' for Docker.")
                        continue
                    container_port = plugin.get("port",80)
                    try:
                        result = subprocess.run(["docker","run","-d","--rm","-P",image],
                                                capture_output=True,text=True)
                        if result.returncode != 0:
                            logger.error(f"Docker launch failed '{name}': {result.stderr.strip()}")
                            continue
                        cid = result.stdout.strip()
                        self._launched_containers.append(cid)
                        logger.info(f"'{name}' Docker container {cid[:12]} started.")
                        port_res = subprocess.run(["docker","port", cid, str(container_port)],
                                                  capture_output=True,text=True)
                        if port_res.returncode!=0 or not port_res.stdout:
                            logger.error(f"No port map for '{name}': {port_res.stderr.strip()}")
                            subprocess.run(["docker","rm","-f",cid])
                            self._launched_containers.remove(cid)
                            continue
                        mapping = port_res.stdout.strip()
                        host_port = None
                        if "->" in mapping:
                            host_port = mapping.split("->")[-1].split(":")[-1]
                        host_port = host_port.strip() if host_port else None
                        if not host_port:
                            logger.error(f"Unrecognized port map '{name}': {mapping}")
                            subprocess.run(["docker","rm","-f",cid])
                            self._launched_containers.remove(cid)
                            continue
                        url = f"http://localhost:{host_port}"
                        logger.info(f"'{name}' Docker SSE at {url} (Pseudo).")
                        self.mcp_servers.append({"server_type":"docker_sse","url":url,"name":name})
                    except FileNotFoundError:
                        logger.error(f"Docker not installed - cannot run '{name}'.")
                        continue
                else:
                    logger.warning(f"'{name}' uses unsupported transport '{transport}'. Will skip or isolate.")
                    if plugin.get("image"):
                        try:
                            result = subprocess.run(["docker","run","-d","--rm",plugin["image"]],
                                                    capture_output=True,text=True)
                            if result.returncode==0:
                                cid = result.stdout.strip()
                                self._launched_containers.append(cid)
                                logger.warning(f"'{name}' isolated in Docker (no integration).")
                            else:
                                logger.error(f"Failed to isolate '{name}': {result.stderr.strip()}")
                        except Exception as e:
                            logger.error(f"Error isolating '{name}': {e}")
                    continue

            except Exception as e:
                logger.exception(f"Error processing plugin '{name}': {e}")

        logger.info(f"Discovery complete. {len(self.mcp_servers)} integrated.")
        if self.mcp_servers:
            names=[str(s.get('name')) for s in self.mcp_servers]
            logger.info("Trusted servers: " + ", ".join(names))

    def _discover_from_github(self) -> List[Dict[str,Any]]:
        discovered=[]
        try:
            idx="https://raw.githubusercontent.com/modelcontextprotocol/servers/main/README.md"
            resp=requests.get(idx,timeout=5)
            if resp.status_code==200:
                content=resp.text
                for line in content.splitlines():
                    if line.startswith("* ["):
                        start=line.find("](https://github.com/")
                        if start!=-1:
                            end=line.find(")",start)
                            if end!=-1:
                                repo_url=line[start+2:end]
                                plugin_name=line[line.find("[")+1:line.find("]")]
                                manifest=self._fetch_repo_manifest(repo_url)
                                if manifest:
                                    manifest["source"]=repo_url
                                    if "name" not in manifest:
                                        manifest["name"]=plugin_name
                                    discovered.append(manifest)
                                else:
                                    image=None
                                    command=None
                                    args=None
                                    if repo_url.endswith(".git"):
                                        repo_url=repo_url[:-4]
                                    parts=repo_url.rstrip("/").split("/")
                                    rname=parts[-1] if parts else ""
                                    if "server" in rname.lower():
                                        command="npx"
                                        args=["-y",f"@modelcontextprotocol/{rname}"]
                                    if command or image:
                                        pl={"name":plugin_name,"source":repo_url}
                                        if command:
                                            pl.update({"transport":"stdio","command":command,"args":args})
                                        if image:
                                            pl.update({"transport":"docker","image":image})
                                        discovered.append(pl)
            else:
                logger.warning(f"GitHub discovery fail status {resp.status_code}")
        except Exception as e:
            logger.error(f"GitHub discovery error: {e}")
        logger.info(f"GitHub discovered {len(discovered)} plugins.")
        return discovered

    def _fetch_repo_manifest(self, repo_url:str)->Optional[Dict[str,Any]]:
        raw_base=repo_url.replace("https://github.com","https://raw.githubusercontent.com")
        if raw_base.endswith("/"):
            raw_base=raw_base[:-1]
        if raw_base.endswith(".git"):
            raw_base=raw_base[:-4]
        for branch in ("main","master"):
            for fname in ("mcp.json","MCP.json","manifest.json","mcp.yaml","MCP.yaml","manifest.yaml"):
                url=f"{raw_base}/{branch}/{fname}"
                try:
                    resp=requests.get(url,timeout=3)
                    if resp.status_code==200:
                        txt=resp.text
                        if fname.endswith(".json"):
                            return json.loads(txt)
                        else:
                            try:
                                import yaml
                                return yaml.safe_load(txt)
                            except:
                                logger.warning(f"YAML parse failed {url}")
                                return None
                except:
                    continue
        return None

    def _discover_from_dnssd(self)->List[Dict[str,Any]]:
        discovered=[]
        if Zeroconf:
            try:
                services=[]
                def on_service_state_change(zc:Zeroconf,stype:str,name:str,sc):
                    if sc is ServiceStateChange.Added:
                        try:
                            info:ServiceInfo=zc.get_service_info(stype,name)
                            if info:
                                host=None
                                if info.addresses:
                                    try:
                                        host=socket.inet_ntoa(info.addresses[0])
                                    except:
                                        try:
                                            host=socket.inet_ntop(socket.AF_INET6,info.addresses[0])
                                        except:
                                            host=None
                                if not host and info.server:
                                    host=info.server.strip(".")
                                port=info.port
                                if host:
                                    url=f"http://{host}:{port}"
                                    plugin={"name":info.name,"transport":"sse","url":url,
                                            "domain":info.server.rstrip('.') if info.server else None,
                                            "source":"mDNS"}
                                    if info.properties:
                                        try:
                                            for k,v in info.properties.items():
                                                kk=k.decode('utf-8') if isinstance(k,bytes) else str(k)
                                                vv=v.decode('utf-8') if isinstance(v,bytes) else str(v)
                                                plugin[kk]=vv
                                        except:
                                            pass
                                    services.append(plugin)
                        except Exception as e:
                            logger.error(f"mDNS service '{name}' error: {e}")
                self._zeroconf=Zeroconf()
                ServiceBrowser(self._zeroconf,"_mcp._tcp.local.",handlers=[on_service_state_change])
                import time
                time.sleep(3)
                self._zeroconf.close()
                self._zeroconf=None
                if services:
                    discovered.extend(services)
            except Exception as e:
                logger.error(f"mDNS discovery error: {e}")
        else:
            logger.debug("No zeroconf, skipping mDNS.")

        if dns and dns.resolver:
            test_domains=[]
            for ref in self.discovery_referrals:
                if isinstance(ref,str) and "." in ref and not ref.startswith("http") and not ref.startswith("_mcp._tcp"):
                    test_domains.append(ref)
            test_domains=list({d.lower() for d in test_domains})
            for domain in test_domains:
                q=f"_mcp._tcp.{domain}"
                try:
                    ans=dns.resolver.resolve(q,"SRV")
                    for a in ans:
                        tgt=str(a.target).rstrip(".")
                        port=a.port
                        url=f"http://{tgt}:{port}"
                        plugin={"transport":"sse","url":url,"domain":tgt,"source":"DNS-SD"}
                        try:
                            txtans=dns.resolver.resolve(q,"TXT")
                            for t in txtans:
                                txt_data="".join(s.decode('utf-8') for s in t.strings)
                                for part in txt_data.split(";"):
                                    if '=' in part:
                                        k,v=part.split('=',1)
                                        plugin[k.strip()]=v.strip()
                        except:
                            pass
                        discovered.append(plugin)
                except Exception as e:
                    logger.debug(f"No SRV for '{q}': {e}")
        else:
            if dns is None:
                logger.debug("dnspython not installed, skipping DNS-SD.")
        if discovered:
            logger.info(f"DNS-SD discovered {len(discovered)} plugin(s).")
        return discovered

    def _process_referrals(self,referrals:List[str])->List[Dict[str,Any]]:
        discovered=[]
        for ref in referrals:
            if not isinstance(ref,str):
                continue
            if ref.startswith("http://") or ref.startswith("https://"):
                try:
                    resp=requests.get(ref,timeout=5)
                    if resp.status_code==200:
                        data=resp.text
                        plugin=None
                        try:
                            plugin=json.loads(data)
                        except json.JSONDecodeError:
                            try:
                                import yaml
                                plugin=yaml.safe_load(data)
                            except:
                                plugin=None
                        if plugin:
                            plugin["source"]=ref
                            discovered.append(plugin)
                            continue
                except Exception as e:
                    logger.debug(f"Referral fetch fail {ref}: {e}")
            if "github.com" in ref or "/" in ref:
                if ref.startswith("http"):
                    gh_url=ref
                else:
                    parts=ref.split("/")
                    if len(parts)==2:
                        owner,repo=parts
                        gh_url=f"https://raw.githubusercontent.com/{owner}/{repo}/main/mcp.json"
                    else:
                        gh_url=None
                if gh_url:
                    try:
                        resp=requests.get(gh_url,timeout=5)
                        if resp.status_code==200:
                            plugin=json.loads(resp.text)
                            plugin["source"]=gh_url
                            discovered.append(plugin)
                            continue
                    except Exception as e:
                        logger.debug(f"GitHub referral fail {gh_url}: {e}")
        return discovered

    def shutdown(self)->None:
        async def _close_server(s):
            try:
                pass
            except Exception as e:
                logger.debug(f"Error closing server {s}: {e}")
        async def _close_all():
            tasks=[asyncio.create_task(_close_server(s)) for s in self.mcp_servers]
            if tasks:
                await asyncio.gather(*tasks,return_exceptions=True)
        try:
            asyncio.run(_close_all())
        except RuntimeError:
            pass
        for cid in list(self._launched_containers):
            try:
                subprocess.run(["docker","rm","-f",cid],timeout=5,
                               stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            except Exception as e:
                logger.warning(f"Failed to stop container {cid}: {e}")
        self._launched_containers.clear()
        logger.info("All plugin servers & containers shut down.")
MCP_PY

###############################################################################
# Minimal .dockerignore
###############################################################################
cat <<'DOCKERIGNORE' > "$TARGET_DIR/.dockerignore"
.git
__pycache__
DOCKERIGNORE

###############################################################################
# requirements.txt (including needed packages to avoid 500 error in /tree_search)
###############################################################################
cat <<'REQS' > "$TARGET_DIR/requirements.txt"
numpy<2.0.0
requests>=2.28.0
pyyaml>=6.0
openai>=0.27.0
anthropic>=0.49.0
chromadb>=0.6.3
weaviate-client>=4.11.0
pandas>=1.5.0

torch==2.0.1
transformers==4.28.0
sentence-transformers==2.2.2

dnspython>=2.2.1
zeroconf>=0.39.0
cryptography>=39.0.0
openai-agents>=0.0.6

flask>=2.2.0
shutup>=0.2.0
humanize>=4.0.0
dataclasses-json>=0.5.7
REQS

###############################################################################
# Dockerfile
###############################################################################
cat <<'DOCKERFILE' > "$TARGET_DIR/Dockerfile"
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./

# Install system packages + cryptography build deps + wheel
RUN apt-get update && \
    apt-get install -y curl build-essential libssl-dev libffi-dev python3-dev python3-wheel && \
    rm -rf /var/lib/apt/lists/* && \
    # Ensuring the newest pip/setuptools/wheel to fix "bdist_wheel not found":
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    # Install Ollama in container
    curl -fsSL https://ollama.com/install.sh | sh && \
    ln -s /usr/bin/ollama /usr/local/bin/ollama || true && \
    # Install Python deps
    pip install --no-cache-dir -r requirements.txt && \
    # Re-install huggingface_hub==0.10.1 for older sentence-transformers
    pip uninstall -y huggingface_hub && \
    pip install --no-cache-dir huggingface_hub==0.10.1 && \
    # Additional forced step so we confirm cryptography + SentenceTransformer + Flask are present
    python -c "import cryptography; import sentence_transformers; import flask" && \
    echo 'export PATH="/usr/local/bin:$PATH"' >> /root/.bashrc

COPY sovereign_agent.py ./
COPY sovereign_alpha_mcp_extension.py ./
COPY trust_whitelist.yaml ./
COPY sovereign_agent_web.html ./sovereign_agent_web.html
COPY agentic_treesearch.py ./

EXPOSE 5000
RUN mkdir -p data
CMD ["python","sovereign_agent.py"]
DOCKERFILE

###############################################################################
# docker-compose.yaml
###############################################################################
cat <<'COMPOSE' > "$TARGET_DIR/docker-compose.yaml"
version: "3.8"
services:
  agent:
    build: .
    image: sovereign-agent:latest
    container_name: sovereign_agent
    network_mode: host
    environment:
      - LLM_PROVIDER=${LLM_PROVIDER-ollama}
      - OPENAI_API_KEY=${OPENAI_API_KEY-}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY-}
      - OLLAMA_MODEL=${OLLAMA_MODEL-gemma3:4b}
      - OLLAMA_URL=${OLLAMA_URL-http://localhost:11434}
      - MEMORY_BACKEND=${MEMORY_BACKEND-chroma}
      - EMBEDDING_MODE=${EMBEDDING_MODE-local}
      - WEAVIATE_URL=${WEAVIATE_URL-http://localhost:8080}
    stdin_open: true
    tty: true
COMPOSE

###############################################################################
# trust_whitelist.yaml
###############################################################################
cat <<'TRUST' > "$TARGET_DIR/trust_whitelist.yaml"
allowed_domains:
  - "example.com"
allowed_paths:
  - "./data"
allowed_tools:
  - "web_fetch"
  - "file_read"
  - "file_write"
allowed_cert_fingerprints: []
allowed_ed25519_keys: []
TRUST

###############################################################################
# README.md
###############################################################################
cat <<'README' > "$TARGET_DIR/README.md"
# Sovereign Agentic AGI ALPHA (v6/v16)

This single-file, production-ready agent integrates chain-of-thought
and ReAct orchestration for advanced reasoning, plus the **Sovereign
Alpha Agent MCP Extension** for secure self-extension, and now
**Agentic Tree Search**.

## Common Fixes
- cryptography installed (>=39.0.0) to fix "cryptography library is required" error.
- pinned sentence-transformers + torch + transformers to avoid "init_empty_weights not defined".
- pinned numpy<2.0.0.
- Minimal Flask-based web UI at http://localhost:5000 or http://<your-host>:5000, 
  serving the big HTML page, so a non-technical user can interact.
- The web chat fetch is now a relative "/chat" call to avoid "Failed to fetch."
- Dockerfile now installs `python3-wheel` so that sentence-transformers==2.2.2
  can build wheels properly in the container.
- We also explicitly upgrade pip, setuptools, and wheel inside Docker to fix "invalid command 'bdist_wheel'".
- Additionally, a `/tree_search` route is exposed to demonstrate "Agentic Tree Search"
  usage from the web side, returning BFS expansions or best path in the JSON.
- **Extra Debug**: we log everything in the `/tree_search` route, so check your container logs.

## Usage
1. `docker-compose build`
2. `docker-compose up -d`
3. The script auto-attaches to `sovereign_agent` for Terminal chat (type exit/quit to end).
4. Meanwhile, the minimal Flask UI is at http://localhost:5000 (or http://<your-host>:5000 if remote).
   - Connect your Phantom wallet, verify tokens, etc.
   - Send messages in the web chat.
   - For advanced demonstration, use the "Alpha Explorer with Agentic Tree Search" 
     section in the webpage or POST JSON to `/tree_search` with `{"topic":"some topic"}`
     to run the BFS-based agentic treesearch logic. The BFS expansions and best node
     are returned in `resultDetails`.
5. `docker-compose logs -f agent` to see logs or `docker attach sovereign_agent`.

If /tree_search returns HTTP 500, the container logs will show a traceback after
**`"Error in /tree_search route"`**. That traceback is critical to diagnosing the error.
README

###############################################################################
# The big webpage (sovereign_agent_web.html) with a small UI to call /tree_search
###############################################################################
cat <<'WEB_HTML' > "$TARGET_DIR/sovereign_agent_web.html"
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>AGI ALPHA AGENT – Official Website</title>
  <!-- LOAD SOLANA WEB3.JS -->
  <script src="https://unpkg.com/@solana/web3.js@1.76.0/lib/index.iife.js"></script>

  <style>
  /* GLOBAL RESETS & FONTS */
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }
  html, body {
    width: 100%;
    height: 100%;
    font-family: "Open Sans", sans-serif;
    overflow-x: hidden;
  }

  body {
    background: linear-gradient(135deg, rgba(255,250,235,1), rgba(255,245,220,1));
    color: #333;
    min-height: 100vh;
    line-height: 1.4;
    scroll-behavior: smooth;
    position: relative;
  }

  #p5-background {
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%; z-index: -1;
    background: url("https://ipfs.io/ipfs/QmQ6MogCaNn5xMWkFa2jtvYSoiTJXtapp3unTenD65SDak")
                center center no-repeat;
    background-size: cover;
    opacity: 0.3;
    mix-blend-mode: multiply;
  }
  #asi-hieroglyphs {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; pointer-events: none;
    background: repeating-linear-gradient(
      160deg,
      transparent 0 10px,
      rgba(255,205,160,0.05) 10px 20px,
      transparent 20px 30px,
      rgba(255,225,200,0.03) 30px 40px
    );
    mix-blend-mode: screen;
  }

  #main-content {
    display: none;
  }

  header {
    width: 100%;
    padding: 3rem 1rem 2rem;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  header .agent-image {
    max-width: 500px; width: 85%; border-radius: 0.5rem;
    box-shadow: 0 0 35px rgba(0,0,0,0.25); margin-bottom: 1.5rem;
  }
  header h1 {
    font-family: "Montserrat", sans-serif; font-weight: 700;
    font-size: 2.2rem; color: #8e6d2a; text-transform: uppercase;
    margin-bottom: 1rem; letter-spacing: 0.08em;
  }
  header p.tagline {
    font-size: 1.15rem; color: #a28154; margin-bottom: 1.5rem;
  }
  header a.cta-button {
    display: inline-block; text-decoration: none; padding: 0.9rem 1.5rem;
    font-size: 1.15rem; font-weight: 600; border-radius: 30px; color: #fff;
    background: linear-gradient(135deg, #ffd28f, #fbb45c);
    transition: all 0.4s ease; box-shadow: 0 0 15px rgba(255,255,255,0.3);
  }
  header a.cta-button:hover {
    box-shadow: 0 0 35px rgba(255,255,255,0.5); transform: scale(1.06);
  }

  main {
    width: 90%; max-width: 1100px; margin: 2rem auto;
    background-color: rgba(255,255,255, 0.8);
    backdrop-filter: blur(5px); border-radius: 0.5rem;
    padding: 2rem; box-shadow: 0 0 25px rgba(255,255,255,0.5);
  }
  main h2 {
    font-family: "Montserrat", sans-serif; font-weight: 700; font-size: 1.7rem; color: #8e6d2a; margin: 1rem 0;
  }
  main .code-block {
    display: inline-block; background: #fff3e0; padding: 0.3rem 0.6rem; border-radius: 4px;
    font-family: monospace; color: #8e6d2a;
  }
  main p {
    margin-bottom: 1.2rem; font-size: 1rem; color: #333; line-height: 1.6;
  }
  main p strong {
    font-weight: 700; color: #8e6d2a;
  }

  section {
    margin-bottom: 2rem; opacity: 0; transform: translateY(30px); transition: all 0.8s ease;
  }
  section.in-view {
    opacity: 1; transform: translateY(0);
  }

  hr {
    margin: 2rem 0; border: none; height: 1px; background-color: rgba(142,109,42,0.25);
  }

  .connect-btn {
    display: inline-block; text-decoration: none; padding: 0.9rem 1.5rem; font-size: 1.05rem;
    font-weight: 600; border-radius: 30px; color: #fff;
    background: linear-gradient(135deg, #ffd28f, #fbb45c);
    transition: all 0.4s ease; box-shadow: 0 0 15px rgba(255,255,255,0.3);
    cursor: pointer;
  }
  .connect-btn:hover {
    box-shadow: 0 0 35px rgba(255,255,255,0.5); transform: scale(1.06);
  }
  #access-message {
    margin-top: 1rem; font-weight: bold; font-size: 1rem; color: #8e6d2a; text-align: center;
  }
  #web-chat {
    margin-top: 2rem; padding: 1rem; border-radius: 5px; background: #fef9f3; box-shadow: 0 0 5px rgba(0,0,0,0.1);
  }
  #chat-log {
    width: 100%; height: 200px; background: #ffffff; overflow-y: auto; margin-bottom: 1rem;
    padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px; font-size: 0.9rem; color: #444;
  }
  #user-input {
    width: 75%; padding: 0.6rem; font-size: 1rem; margin-right: 0.5rem;
  }
  #send-btn {
    padding: 0.6rem 1rem; font-size: 1rem; font-weight: 600; color: #fff; background: #fbb45c;
    border: none; border-radius: 4px; cursor: pointer; box-shadow: 0 0 3px rgba(0,0,0,0.1);
  }
  #send-btn:hover {
    background: #f7a53d;
  }

  @media (max-width: 768px) {
    header .agent-image { max-width: 300px; }
    header h1 { font-size: 1.6rem; margin-bottom: 0.7rem; }
    header p.tagline { font-size: 1rem; }
    header a.cta-button { font-size: 1rem; }
    main { padding: 1.5rem; }
    #user-input {
      width: 100%; margin-bottom: 0.5rem;
    }
    #send-btn {
      width: 100%; margin-bottom: 1rem;
    }
  }
  </style>
</head>

<body>
<div id="p5-background"></div>
<div id="asi-hieroglyphs"></div>

<div style="text-align:center; margin-top:3rem;">
  <button id="connect-page" class="connect-btn">Connect Wallet (500k+)</button>
  <p id="page-access-message"></p>
</div>

<div id="main-content">
  <header>
    <img class="agent-image"
         src="https://ipfs.io/ipfs/QmQ6MogCaNn5xMWkFa2jtvYSoiTJXtapp3unTenD65SDak"
         alt="AGI ALPHA AGENT">
    <h1>AGI ALPHA AGENT</h1>
    <p class="tagline">The Official AGI ALPHA AGENT Website</p>
    <a class="cta-button"
       href="https://dexscreener.com/solana/8zq3vBuoy66dur6dhrA4aqnrtGg9yZyRAp51BTBpEXj"
       target="_blank"
       rel="noopener">
      Buy $AGIALPHA Tokens
    </a>
    <p style="margin-top:1rem; font-size:0.95rem; color:#8e6d2a;">
      Contract Address (CA):
      <strong>tWKHzXd5PRmxTF5cMfJkm2Ua3TcjwNNoSRUqx6Apump</strong>
    </p>
  </header>

  <main>
    <section id="web-chat">
      <h2>Chat with AGI ALPHA AGENT (Web)</h2>
      <div id="chat-log"></div>
      <input type="text" id="user-input" placeholder="Type your message here..."/>
      <button id="send-btn">Send</button>
      <p style="font-size:0.85rem; color:#666; margin-top:0.5rem;">
        This sends a POST request to <code>/chat</code> on the same origin 
        (so if you're accessing <code>http://192.168.x.x:5000</code>, it calls <code>http://192.168.x.x:5000/chat</code>).
      </p>
    </section>

    <hr>

    <section id="agentic-tree-search">
      <h2>Alpha Explorer with Agentic Tree Search</h2>
      <p>Enter a "topic" or "signal" to explore, then press "Explore" to run the agentic tree search. This uses the advanced "Agentic Tree Search" engine to discover potential alpha signals in an open-ended environment. (Demo)</p>
      <div style="display:flex; flex-wrap:wrap; align-items:center;">
        <input type="text" id="alpha-topic" placeholder="Enter a topic or signal"
               style="flex:1; min-width:200px; padding:0.5rem; margin-right:0.5rem;"/>
        <button id="explore-btn"
                style="padding: 0.6rem 1rem; font-size: 1rem; font-weight: 600; background: #fbb45c; border:none; border-radius:4px; cursor:pointer;">
          Explore
        </button>
      </div>
      <div id="explore-result"
           style="margin-top:1rem; color:#444; background:#ffffff; padding:0.5rem; border-radius:4px; border:1px solid #ddd; min-height:50px;">
      </div>
    </section>

    <hr>

    <section id="about">
      <h2>About AGI ALPHA AGENT</h2>
      <p>
        <strong>AGI-Alpha-Agent-v0</strong><br>
        <strong>CA:</strong>
        <span class="code-block">tWKHzXd5PRmxTF5cMfJkm2Ua3TcjwNNoSRUqx6Apump</span><br>
        <strong>AGI ALPHA AGENT (ALPHA.AGENT.AGI.Eth) Powered by $AGIALPHA</strong><br>
        <strong>Seize the Alpha. Transform the world.</strong>
      </p>
      <p>
        <a href="https://www.linkedin.com/in/montrealai/" target="_blank" rel="noopener">Vincent Boucher</a>,
        an AI pioneer and President of
        <a href="https://www.montreal.ai/" target="_blank" rel="noopener">MONTREAL.AI</a>
        and <a href="https://www.quebec.ai/" target="_blank" rel="noopener">QUEBEC.AI</a> since 2003,
        reshaped the landscape by dominating the
        <a href="https://web.archive.org/web/20170929214241/https://gym.openai.com/read-only.html"
           target="_blank" rel="noopener">OpenAI Gym</a>
        with <strong>AI Agents</strong> in 2016 (#1 worldwide) and unveiling the game-changing
        <a href="https://www.quebecartificialintelligence.com/priorart"
           target="_blank" rel="noopener">“Multi-Agent AI DAO”</a> blueprint in 2017
        (“<em>The Holy Grail of Foundational IP at the Intersection of AI Agents and Blockchain</em>”).
      </p>
      <p>
        Our <strong>AGI ALPHA AGENT</strong>, fueled by the strictly-utility
        <strong>$AGIALPHA</strong> token, now harnesses that visionary foundation—
        <em>arguably the world’s most valuable, impactful and important IP</em>—to unleash
        the ultimate alpha signal engine.
      </p>
    </section>

    <hr>

    <section id="access-gate">
      <h2>Access the AGI ALPHA AGENT</h2>
      <p>You must hold at least <strong>1,000,000</strong> $AGIALPHA tokens to proceed.</p>
      <button id="connect-wallet" class="connect-btn">Connect Wallet</button>
      <p id="access-message"></p>
      <p>
        <strong>CA:</strong>
        <span class="code-block">tWKHzXd5PRmxTF5cMfJkm2Ua3TcjwNNoSRUqx6Apump</span><br>
      </p>
      <p><strong>Exclusive Access with $AGIALPHA Token</strong></p>
    </section>

    <hr>

    <section id="disclaimers">
      <h2>Initial Terms & Conditions</h2>
      <p><strong>The Emergence of an AGI-Powered Alpha Agent.</strong></p>
      <p><strong>Ticker ($): AGIALPHA</strong></p>
      <p>
        Rooted in the publicly disclosed 2017 “Multi-Agent AI DAO” prior art, the AGI ALPHA AGENT
        utilizes $AGIALPHA tokens purely as utility tokens—no equity, no profit-sharing—intended
        for the purchase of products/services by the AGI ALPHA AGENT (ALPHA.AGENT.AGI.Eth).
        They are not intended for investment or speculative purposes.
      </p>
      <p><strong>1.</strong> <strong>Token Usage</strong>: strictly utility tokens—no equity, no profit-sharing.</p>
      <p><strong>2.</strong> <strong>Non-Refundable</strong>.</p>
      <p><strong>3.</strong> <strong>No Guarantee of Value</strong>.</p>
      <p><strong>4.</strong> <strong>Regulatory Compliance</strong>.</p>
      <p><strong>5.</strong> <strong>User Responsibility</strong>.</p>
      <p><strong>OVERRIDING AUTHORITY:</strong> AGI.Eth</p>
      <p>$AGIALPHA is experimental. Any expectation of profit is unjustified.</p>
      <p>By using $AGIALPHA, you agree to the $AGIALPHA Terms and Conditions.</p>
    </section>

    <hr>

    <section id="further-info">
      <h2>Further Information</h2>
      <p>
        <strong>Discord:</strong>
        <a href="https://discord.gg/montrealai" target="_blank" rel="noopener">https://discord.gg/montrealai</a>
      </p>
      <p>
        <strong>LinkedIn:</strong>
        <a href="https://www.linkedin.com/in/montrealai/" target="_blank" rel="noopener">
          https://www.linkedin.com/in/montrealai/
        </a>
      </p>
      <p>
        <strong>X (AGI ALPHA AGENT):</strong>
        <a href="https://x.com/agialphaagent" target="_blank" rel="noopener">https://x.com/agialphaagent</a>
      </p>
      <p>
        <strong>X (MONTREAL.AI):</strong>
        <a href="https://x.com/Montreal_AI" target="_blank" rel="noopener">https://x.com/Montreal_AI</a>
      </p>
      <p>
        <strong>Facebook Page:</strong>
        <a href="https://www.facebook.com/MontrealAI/" target="_blank" rel="noopener">
          https://www.facebook.com/MontrealAI/
        </a>
      </p>
      <p>
        <strong>Facebook Group:</strong>
        <a href="https://www.facebook.com/groups/MontrealAI" target="_blank" rel="noopener">
          https://www.facebook.com/groups/MontrealAI
        </a>
      </p>
      <p>
        <strong>Telegram:</strong>
        <a href="https://t.me/agialpha" target="_blank" rel="noopener">https://t.me/agialpha</a>
      </p>
      <p>
        <strong>YouTube:</strong>
        <a href="https://www.youtube.com/montrealai" target="_blank" rel="noopener">
          https://www.youtube.com/montrealai
        </a>
      </p>
      <p>
        <strong>Official info about $AGIALPHA</strong> is in the on-chain records of the AGI ALPHA Agent:
        <a href="https://app.ens.domains/alpha.agent.agi.eth" target="_blank" rel="noopener">
          https://app.ens.domains/alpha.agent.agi.eth
        </a>.
      </p>
      <p>Pre-Alpha Version — Under Development.</p>
    </section>
  </main>
</div>

<!-- p5.js: background star effect -->
<script src="https://cdn.jsdelivr.net/npm/p5@1.6.0/lib/p5.min.js"></script>
<script>
let starCount = 33;
let stars = [];
let containerRadius = 300;
let angle = 0;

function setup() {
  const cnv = createCanvas(windowWidth, windowHeight, WEBGL);
  cnv.parent("p5-background");
  colorMode(HSB, 360, 100, 100);
  noStroke();
  for (let i = 0; i < starCount; i++) {
    stars.push(new Star());
  }
}

function draw() {
  background(0, 0.08);
  rotateY(angle * 0.001);
  rotateX(angle * 0.0007);
  angle += 0.2;

  let zoom = sin(frameCount * 0.002) * 150;
  translate(0, 0, zoom);

  push();
  noFill();
  stroke(50, 0, 100, 20);
  sphere(containerRadius);
  pop();

  for (let s of stars) {
    s.update();
    s.render();
  }
}

class Star {
  constructor() {
    let r = random(containerRadius);
    let theta = random(TWO_PI);
    let phi = random(PI);
    this.x = r * sin(phi) * cos(theta);
    this.y = r * sin(phi) * sin(theta);
    this.z = r * cos(phi);
    this.vx = random(-1,1);
    this.vy = random(-1,1);
    this.vz = random(-1,1);
    this.hue = random(360);
    this.size = random(4,7);
  }

  update() {
    this.x += this.vx;
    this.y += this.vy;
    this.z += this.vz;
    let magPos = sqrt(this.x*this.x + this.y*this.y + this.z*this.z);
    if(magPos > containerRadius - this.size) {
      let nx = this.x / magPos;
      let ny = this.y / magPos;
      let nz = this.z / magPos;
      let dot = this.vx*nx + this.vy*ny + this.vz*nz;
      this.vx -= 2*dot*nx;
      this.vy -= 2*dot*ny;
      this.vz -= 2*dot*nz;
    }
  }

  render() {
    push();
    translate(this.x, this.y, this.z);
    fill(this.hue, 80, 100);
    sphere(this.size);
    pop();
    this.hue += 0.4;
    if(this.hue >= 360) this.hue = 0;
  }
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}
</script>

<script>
document.addEventListener("DOMContentLoaded", () => {
  const sections = document.querySelectorAll("section");
  const observerOpts = { root: null, rootMargin: "0px", threshold: 0.1 };
  const observer = new IntersectionObserver((entries, obs) => {
    entries.forEach(entry => {
      if(entry.isIntersecting) {
        entry.target.classList.add("in-view");
        obs.unobserve(entry.target);
      }
    });
  }, observerOpts);
  sections.forEach(sec => observer.observe(sec));
});
</script>

<!-- WALLET CONNECT + BALANCE CHECK SCRIPT -->
<script>
const TOKEN_MINT = "tWKHzXd5PRmxTF5cMfJkm2Ua3TcjwNNoSRUqx6Apump";
const MIN_TOKENS_FOR_PAGE = 500000;
const MIN_TOKENS_FOR_AGENT = 1000000;
const RPC_URL = "https://solana-mainnet.g.alchemy.com/v2/KT1RV46Eoje4kL-SC2pSANHTVm6-wgwF";
const { Connection, PublicKey } = solanaWeb3;
const pageBtn   = document.getElementById("connect-page");
const pageMsg   = document.getElementById("page-access-message");
const agentBtn  = document.getElementById("connect-wallet");
const agentMsg  = document.getElementById("access-message");
const mainEl    = document.getElementById("main-content");

mainEl.style.display = "none";

async function getBalance(ownerPubkey) {
  const connection = new Connection(RPC_URL, "confirmed");
  const tokenAccounts = await connection.getParsedTokenAccountsByOwner(
    new PublicKey(ownerPubkey),
    { programId: new PublicKey("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA") }
  );
  let balance = 0;
  for(const acc of tokenAccounts.value) {
    const info = acc.account.data.parsed.info;
    if(info.mint === TOKEN_MINT) {
      balance += info.tokenAmount.uiAmount;
    }
  }
  return balance;
}

// #1: Connect for PAGE Access (>=500k)
pageBtn.addEventListener("click", async () => {
  if(!window.solana || !window.solana.isPhantom) {
    alert("Phantom Wallet not found! Please install or enable it first.");
    return;
  }
  try {
    await window.solana.connect();
    const pubkeyStr = window.solana.publicKey.toString();
    pageMsg.innerText = `Connected: ${pubkeyStr}\nChecking balance...`;

    const bal = await getBalance(pubkeyStr);
    if(bal >= MIN_TOKENS_FOR_PAGE) {
      pageMsg.innerText = `✅ You hold ${bal.toLocaleString()} $AGIALPHA.\nPage access granted.`;
      mainEl.style.display = "block";
    } else {
      pageMsg.innerText = `⛔ You hold ${bal.toLocaleString()} $AGIALPHA, below 500,000.\nAccess denied.`;
      mainEl.style.display = "none";
    }
  } catch(e) {
    console.error(e);
    alert("Failed to connect or check balance (page).");
  }
});

// #2: Connect for AGENT Access (>=1M)
agentBtn.addEventListener("click", async () => {
  if(!window.solana || !window.solana.isPhantom) {
    alert("Phantom Wallet not found! Please install or enable it first.");
    return;
  }
  try {
    await window.solana.connect();
    const pubkeyStr = window.solana.publicKey.toString();
    agentMsg.innerText = `Connected wallet: ${pubkeyStr}\nChecking if >=1M $AGIALPHA...`;

    const bal = await getBalance(pubkeyStr);
    if(bal >= MIN_TOKENS_FOR_AGENT) {
      agentMsg.innerText = `🚀 Congratulations! You hold ${bal.toLocaleString()} $AGIALPHA.\nAGI ALPHA AGENT Access Granted.`;
    } else {
      agentMsg.innerText = `⛔ You hold ${bal.toLocaleString()} $AGIALPHA, below 1,000,000.\nAccess Denied.`;
    }
  } catch(e) {
    console.error(e);
    alert("Failed to connect or check balance (agent).");
  }
});

// Minimal Web Chat with the container's Flask server on port 5000,
// using relative "/chat" so it works for any IP/host.
const chatLog = document.getElementById("chat-log");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

sendBtn.addEventListener("click", async () => {
  const msg = userInput.value.trim();
  if(!msg) return;
  userInput.value = "";
  chatLog.innerHTML += `<div><strong>You:</strong> ${msg}</div>`;
  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg })
    });
    if(!resp.ok) {
      chatLog.innerHTML += `<div style="color:red;">[HTTP error ${resp.status}]</div>`;
      return;
    }
    const data = await resp.json();
    const reply = data.reply || "[No reply]";
    chatLog.innerHTML += `<div><strong>Agent:</strong> ${reply}</div>`;
    chatLog.scrollTop = chatLog.scrollHeight;
  } catch(err) {
    console.error(err);
    chatLog.innerHTML += `<div style="color:red;">[Error contacting agent: ${err}]</div>`;
  }
});

// Agentic Tree Search UI
const exploreBtn = document.getElementById("explore-btn");
const alphaTopic = document.getElementById("alpha-topic");
const exploreResult = document.getElementById("explore-result");

exploreBtn.addEventListener("click", async () => {
  const topic = alphaTopic.value.trim();
  if(!topic) {
    exploreResult.innerHTML = "<span style='color:red;'>Please enter a topic or signal.</span>";
    return;
  }
  exploreResult.innerText = "Exploring with topic: " + topic + "...";
  try {
    const resp = await fetch("/tree_search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic })
    });
    if(!resp.ok) {
      exploreResult.innerHTML = `<span style="color:red;">[Error: HTTP ${resp.status}]</span>`;
      return;
    }
    const data = await resp.json();
    if (data.resultDetails) {
      exploreResult.innerText = JSON.stringify(data.resultDetails, null, 2);
    } else if (data.result){
      exploreResult.innerText = data.result;
    } else {
      exploreResult.innerText = "[No known result]";
    }
  } catch(err) {
    console.error(err);
    exploreResult.innerHTML = `<span style="color:red;">[Error contacting /tree_search: ${err}]</span>`;
  }
});
</script>

</body>
</html>
WEB_HTML

###############################################################################
# 8) Also create the "agentic_treesearch.py" file
###############################################################################
cat <<'TREES_PY' > "$TARGET_DIR/agentic_treesearch.py"
# agentic_treesearch.py
#
# A minimal stub or placeholder for the Agentic Tree Search engine.
# This file holds advanced BFS / BFTS / custom treesearch logic,
# integrated as a demonstration for open-ended alpha exploration
# from the "/tree_search" endpoint in sovereign_agent.py.
#
# Now returns BFS details (expanded nodes, best node, etc.)
# so the front-end can display them.

import os, re, json, time, pickle, uuid, random, queue, signal, shutil, logging, traceback
import humanize, yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Callable, Set
from enum import Enum, auto
import copy
from multiprocessing import Process, Queue
import pandas as pd

logger = logging.getLogger(__name__)

def perform_experiments_bfts(config_path:str):
    logger.info(f"Perform experiments with config {config_path}")
    # Just a placeholder "BFS expansions" for demonstration:
    expansions = [
        {"node_id": "root", "metric": 0.5},
        {"node_id": "child_1", "metric": 0.7},
        {"node_id": "child_2", "metric": 0.9},
    ]
    best_node = expansions[-1]

    # We'll pretend there's more BFS logic here ...
    # For now, just log them:
    logger.info(f"BFS expansions: {expansions}")
    logger.info(f"Best node: {best_node}")

    # Return them so sovereign_agent.py can pass them to the front-end
    bfs_details = {
        "expansions": expansions,
        "best_node": best_node
    }

    # In reality, you'd parse the real BFS states, final paths, etc. from the agent's logic.
    # For now, we mimic it.
    return bfs_details
TREES_PY

###############################################################################
# Optionally copy local data for persistence
###############################################################################
if [ -d "weaviate_data" ]; then
  mkdir -p "$TARGET_DIR/weaviate_data"
  cp -r weaviate_data/* "$TARGET_DIR/weaviate_data" || true
  echo "Copied weaviate_data/"
fi
if [ -d "chroma_storage" ]; then
  mkdir -p "$TARGET_DIR/chroma_storage"
  cp -r chroma_storage/* "$TARGET_DIR/chroma_storage" || true
  echo "Copied chroma_storage/"
fi
if [ -d "data" ]; then
  mkdir -p "$TARGET_DIR/data"
  cp -r data/* "$TARGET_DIR/data" || true
  echo "Copied data/"
fi

echo "[3/5] Generated files in '$TARGET_DIR'."

###############################################################################
# 6) Build Docker
###############################################################################
cd "$TARGET_DIR"
echo "[4/5] docker-compose build..."
docker-compose build --no-cache

###############################################################################
# 7) docker-compose up -d
###############################################################################
echo "[5/5] Starting containers in background..."
docker-compose up -d

echo
echo "==========================================================="
echo "Sovereign Agentic AGI ALPHA is deployed in '$TARGET_DIR'!"
echo "Containers running in background."
echo
echo "Attaching to sovereign_agent container for a live chat..."
echo "==========================================================="
echo

if command -v docker &> /dev/null; then
  echo "Showing logs (docker-compose logs -f agent) in parallel. (Ctrl+C to stop logs only)"
  docker-compose logs -f agent &
  LOGS_PID=$!
  docker attach sovereign_agent || true
  echo "Container attach ended. Stopping log tail..."
  kill $LOGS_PID 2>/dev/null || true
else
  echo "Docker not found in PATH. Please run 'docker attach sovereign_agent' manually."
fi

echo
echo "==========================================================="
echo "If you typed 'exit' or 'quit' from inside container's chat,"
echo "the agent loop has ended, but container may still be running."
echo "Run 'docker-compose down' to stop it fully."
echo "==========================================================="

echo
echo "Below is the final webpage (with 500,000 token gate, user-friendly wallet connect, etc.),"
echo "and also featuring the new 'Alpha Explorer' to demonstrate Agentic Tree Search."
echo "Note: This exact HTML is also served at http://localhost:5000/ by the container."
echo

cat sovereign_agent_web.html

