# autoresearch-local-llm

Run [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) with a **local LLM** instead of Claude Code. Zero API cost. Single GPU. Fully autonomous.

## What This Is

Autoresearch is an experiment where an LLM autonomously modifies a GPT training script, runs 5-minute experiments, keeps what improves val_bpb, and discards what doesn't. The original uses Claude Code (cloud API) as the researcher.

**This fork replaces Claude Code with Qwen 3.5 9B running locally via ollama.** The LLM and training share the same GPU. No API keys, no cloud dependencies, no per-experiment cost.

## What Changed

| Component | Original | This Fork |
|-----------|----------|-----------|
| AI Researcher | Claude Code (cloud API) | Qwen 3.5 9B via ollama (local) |
| Cost per experiment | API tokens | $0 |
| Depth | 8 layers | 4 layers |
| Device batch size | 128 | 64 |
| Total batch tokens | 524K | 65K |
| Window pattern | SSSL | L |

Model size is reduced because the LLM agent (~12GB VRAM) and training share the same GPU. The agent compensates by running more experiments.

## Files

| File | Purpose |
|------|---------|
| `agent.py` | Local LLM agent — replaces Claude Code in the autoresearch loop |
| `train.py` | GPT training script (modified hyperparameters for shared VRAM) |
| `prepare.py` | Data preparation (unchanged from original) |
| `program.md` | Experiment instructions for the agent |
| `run_pipeline.sh` | Orchestrator: prepare data, create branch, start agent |
| `nosana_setup.sh` | Container bootstrap for Nosana GPU deployment |
| `job.json` | Nosana job definition |

## How It Works

1. **ollama** serves Qwen 3.5 9B locally on the GPU (~12GB VRAM)
2. **agent.py** reads `train.py` and experiment history, asks Qwen to propose a modification
3. Qwen outputs a modified `train.py`
4. Agent validates syntax, git commits, runs `uv run train.py` (5-min experiment)
5. If val_bpb improved — keep. If not — git reset.
6. Loop forever.

```
GPU (48GB VRAM)
├── Qwen 3.5 9B via ollama (~12GB)
└── GPT training via train.py (~35GB)
    ├── Propose modification
    ├── Validate syntax
    ├── Run 5-min experiment
    ├── Keep if val_bpb improved
    └── Discard if not → loop
```

## Deploy on Nosana

### Option 1: Dashboard

1. Go to [nosana.io](https://nosana.io) dashboard
2. Create a new deployment, select **NVIDIA Pro 6000 (SOC2)**
3. Click Configure and paste the contents of `job.json`
4. Create Deployment

### Option 2: CLI

```bash
nosana job post --file job.json --market nvidia-pro6000 --timeout 480 --wait
```

## Run Locally (if you have a GPU)

```bash
# Install ollama and pull the model
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen3.5:9b

# Clone and setup
git clone https://github.com/SohniSwatantra/autoresearch-local-llm.git
cd autoresearch-local-llm
pip install uv
uv sync

# Run
bash run_pipeline.sh
```

Requires a GPU with at least 24GB VRAM (48GB recommended for full-size experiments).

## Cost

| Setup | Cost per experiment | 100 experiments |
|-------|-------------------|-----------------|
| Original (Claude Code API) | ~$0.05-0.20 | $5-20 |
| This fork (Nosana Pro 6000) | $0.08 (5min at $1/hr) | ~$8 total |
| This fork (own GPU) | $0 | $0 |

## Configuration

Edit `agent.py` to change the local LLM:

```python
MODEL = "qwen3.5:9b"  # Any ollama model works
```

Edit `train.py` hyperparameters to adjust for your GPU's available VRAM:

```python
DEPTH = 4              # Increase if you have more VRAM
DEVICE_BATCH_SIZE = 64 # Increase if you have more VRAM
TOTAL_BATCH_SIZE = 2**16
```

## Credits

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — original framework
- [Qwen 3.5](https://github.com/QwenLM/Qwen3) — local LLM
- [ollama](https://ollama.com) — local LLM serving
- [Nosana](https://nosana.com/?utm_source=reddit&utm_medium=community) - decentralized GPU compute
