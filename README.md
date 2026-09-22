# quantic-hr-ai-rag-group-project

Group project: a Retrieval-Augmented Generation (RAG) app for Human Resources content.

**Stack:** Python 3.12 · [uv](https://docs.astral.sh/uv/) · LangChain / LangGraph · ChromaDB · sentence-transformers

## Setup

### 1. Install uv

macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If your system doesn't have `curl`, you can use `wget`:

```bash
wget-qO-https://astral.sh/uv/install.sh|sh
```




Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal, then confirm with `uv --version`. Other install options: https://docs.astral.sh/uv/getting-started/installation/

### 2. Clone the repo and install dependencies

```bash
git clone https://github.com/Rob-Ottogalli/quantic-hr-ai-rag-group-project.git
cd quantic-hr-ai-rag-group-project
uv sync
```

`uv sync` creates a `.venv`, installs the Python version in `.python-version` if needed, and installs the exact package versions pinned in `uv.lock`.

### 3. Configure environment variables

```bash
cp .env.example .env        # Windows PowerShell: Copy-Item .env.example .env
```

Fill in your own API key(s) in `.env`. This file is git-ignored; never commit it.

### 4. Run things

```bash
uv run python main.py
```

`uv run` uses the project environment automatically, so there is no need to activate `.venv`.

## Day-to-day workflow

- **After every `git pull`, run `uv sync`** so your environment matches any dependency changes teammates made.
- **Add a dependency with `uv add <package>`** (not `pip install`), then commit both `pyproject.toml` and `uv.lock`.
- **Continuous integration:** on every push and pull request to `main`, GitHub Actions runs `uv sync --locked` on Windows and macOS and checks that the core libraries import. A red check usually means `uv.lock` is out of sync with `pyproject.toml`.

## Data

- **`corpus/`** — the RAG source corpus: 12 HR policy documents for the fictional company Northbridge Analytics (PTO, remote work, tax/location, expenses, equipment, infosec, benefits, leave, onboarding, conduct, HR ops), in markdown/HTML/plain-text with numbered sections (`PTO-3`, `RW-4`, etc.) for citation. See `corpus/README.md` for the document index and ingestion conventions.
- **`mock_data/`** — synthetic structured HR data the agent's MCP tools should read from: employees, PTO balances, benefits elections, offices, HR tickets, expense claims, and PTO blackout periods. See `mock_data/README.md` for schemas and the specific edge cases built in for evaluation (e.g. an employee with an insufficient PTO balance, an international remote-work request that exceeds policy limits).
