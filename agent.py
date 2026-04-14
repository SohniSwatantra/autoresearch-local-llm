"""
Local LLM Agent for autoresearch — replaces Claude Code with a local LLM via ollama.

Uses search/replace blocks instead of full file rewrites for faster, more reliable
modifications. Proposes modifications to train.py, validates syntax, runs experiments,
and keeps/discards based on val_bpb.

Usage: PYTHONUNBUFFERED=1 uv run python3 agent.py
"""

import os
import re
import ast
import sys
import time
import subprocess
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = os.environ.get("AUTORESEARCH_MODEL", "qwen3:8b")
TRAIN_SCRIPT = "train.py"
RESULTS_FILE = "results.tsv"
RUN_LOG = "run.log"
RUN_TIMEOUT = 900  # 15 minutes max per experiment
MAX_CONSECUTIVE_CRASHES = 3

# ---------------------------------------------------------------------------
# Ollama interface
# ---------------------------------------------------------------------------

def query_llm(prompt, max_tokens=4096):
    """Query the local LLM via ollama. Returns the response text."""
    import requests
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7,
            }
        }, timeout=600)
        if resp.ok:
            return resp.json().get("response", "")
        else:
            print(f"Ollama error: {resp.status_code} {resp.text[:200]}")
            return ""
    except Exception as e:
        print(f"Ollama connection error: {e}")
        return ""


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git_run(*args):
    """Run a git command, return stdout."""
    result = subprocess.run(["git"] + list(args), capture_output=True, text=True, timeout=30)
    return result.stdout.strip()


def git_commit(message):
    """Stage train.py and commit."""
    subprocess.run(["git", "add", TRAIN_SCRIPT], check=True, timeout=30)
    subprocess.run(["git", "commit", "-m", message], check=True, timeout=30)
    return git_run("rev-parse", "--short", "HEAD")


def git_reset_hard(commit):
    """Reset to a specific commit."""
    subprocess.run(["git", "reset", "--hard", commit], check=True, timeout=30)


def get_current_commit():
    """Get current short commit hash."""
    return git_run("rev-parse", "--short", "HEAD")


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment():
    """Run train.py and return (val_bpb, peak_vram_mb) or (None, None) on failure."""
    print(f"  Running experiment... (timeout: {RUN_TIMEOUT}s)")
    try:
        with open(RUN_LOG, "w") as log_file:
            proc = subprocess.run(
                ["uv", "run", TRAIN_SCRIPT],
                stdout=log_file, stderr=subprocess.STDOUT,
                timeout=RUN_TIMEOUT,
            )
        if proc.returncode != 0:
            print(f"  Experiment crashed (exit code {proc.returncode})")
            return None, None
    except subprocess.TimeoutExpired:
        print(f"  Experiment timed out after {RUN_TIMEOUT}s")
        return None, None
    except Exception as e:
        print(f"  Experiment error: {e}")
        return None, None

    # Parse results
    val_bpb = None
    peak_vram = None
    try:
        with open(RUN_LOG, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("val_bpb:"):
                    val_bpb = float(line.split(":")[1].strip())
                elif line.startswith("peak_vram_mb:"):
                    peak_vram = float(line.split(":")[1].strip())
    except Exception as e:
        print(f"  Error parsing run.log: {e}")

    return val_bpb, peak_vram


def get_crash_info():
    """Get last 50 lines of run.log for crash diagnosis."""
    try:
        with open(RUN_LOG, "r") as f:
            lines = f.readlines()
        return "".join(lines[-50:])
    except Exception:
        return "Could not read run.log"


# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------

def init_results():
    """Initialize results.tsv if it doesn't exist."""
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")


def log_result(commit, val_bpb, memory_gb, status, description):
    """Append a result to results.tsv."""
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n")
    print(f"  Logged: {commit} | val_bpb={val_bpb:.6f} | {memory_gb:.1f}GB | {status} | {description}")


def get_results_history():
    """Read results.tsv and return as string."""
    if not os.path.exists(RESULTS_FILE):
        return "No results yet."
    with open(RESULTS_FILE, "r") as f:
        return f.read()


def get_best_bpb():
    """Get the best (lowest) val_bpb from results history."""
    best = float("inf")
    if not os.path.exists(RESULTS_FILE):
        return best
    with open(RESULTS_FILE, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4 and parts[3] == "keep":
                try:
                    bpb = float(parts[1])
                    if bpb > 0 and bpb < best:
                        best = bpb
                except ValueError:
                    continue
    return best


# ---------------------------------------------------------------------------
# Code modification (search/replace based)
# ---------------------------------------------------------------------------

def read_train_py():
    """Read current train.py."""
    with open(TRAIN_SCRIPT, "r") as f:
        return f.read()


def write_train_py(code):
    """Write modified train.py."""
    with open(TRAIN_SCRIPT, "w") as f:
        f.write(code)


def validate_syntax(code):
    """Check if Python code is syntactically valid."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, str(e)


def extract_hyperparams(code):
    """Extract the hyperparameter section from train.py for the prompt."""
    lines = code.split("\n")
    hyper_lines = []
    in_section = False
    for line in lines:
        if "Hyperparameters" in line or "# Model architecture" in line:
            in_section = True
        if in_section and ("Setup:" in line or "tokenizer" in line.lower()):
            break
        if in_section:
            hyper_lines.append(line)
    return "\n".join(hyper_lines) if hyper_lines else ""


def extract_model_section(code):
    """Extract model architecture section."""
    lines = code.split("\n")
    model_lines = []
    in_section = False
    for i, line in enumerate(lines):
        if "GPT Model" in line or "class GPTConfig" in line:
            in_section = True
        if in_section:
            model_lines.append(line)
        if in_section and line.strip().startswith("class MuonAdamW"):
            break
    return "\n".join(model_lines) if model_lines else ""


def apply_search_replace(code, search, replace):
    """Apply a single search/replace operation. Returns (new_code, success)."""
    if search in code:
        return code.replace(search, replace, 1), True
    # Try with stripped whitespace matching
    search_stripped = "\n".join(l.rstrip() for l in search.split("\n"))
    code_stripped = "\n".join(l.rstrip() for l in code.split("\n"))
    if search_stripped in code_stripped:
        return code_stripped.replace(search_stripped, replace, 1), True
    # Try matching just the key lines (skip comment-only lines in search)
    search_code_lines = [l for l in search.split("\n") if l.strip() and not l.strip().startswith("#")]
    if search_code_lines:
        first_line = search_code_lines[0].rstrip()
        last_line = search_code_lines[-1].rstrip()
        code_lines = code.split("\n")
        for i, line in enumerate(code_lines):
            if line.rstrip() == first_line:
                for j in range(i, min(i + len(search.split("\n")) + 5, len(code_lines))):
                    if code_lines[j].rstrip() == last_line:
                        # Validate match span is close to expected length
                        expected_lines = len([ln for ln in search.split("\n") if ln.strip()])
                        actual_lines = len([ln for ln in code_lines[i:j+1] if ln.strip()])
                        if abs(actual_lines - expected_lines) > 3:
                            continue
                        original_block = "\n".join(code_lines[i:j+1])
                        return code.replace(original_block, replace, 1), True
    return code, False


def parse_search_replace_blocks(response):
    """Parse SEARCH/REPLACE blocks from LLM response.

    Expected format:
    <<<SEARCH
    old code here
    >>>
    <<<REPLACE
    new code here
    >>>
    """
    # Strip markdown code fences the LLM may wrap around the blocks
    cleaned = re.sub(r'```[a-zA-Z]*\n', '', response)
    cleaned = re.sub(r'\n```', '', cleaned)

    blocks = []
    # Accept both formats: with or without >>> between SEARCH and REPLACE
    pattern = r'<<<SEARCH\n(.*?)(?:>>>\s*)?<<<REPLACE\n(.*?)>>>'
    matches = re.findall(pattern, cleaned, re.DOTALL)
    for search, replace in matches:
        blocks.append((search.rstrip("\n"), replace.rstrip("\n")))
    return blocks


# ---------------------------------------------------------------------------
# Agent prompts
# ---------------------------------------------------------------------------

def extract_hyperparams_as_lines(code):
    """Extract all top-level hyperparameter assignments as a clean list."""
    lines = []
    in_section = False
    for line in code.split("\n"):
        if "# Model architecture" in line or "# Optimization" in line or "# Model size" in line:
            in_section = True
        if in_section and line.startswith("# ---"):
            in_section = False
        if in_section and re.match(r'^[A-Z][A-Z_0-9]*\s*=', line):
            lines.append(line)
    return "\n".join(lines)


def build_experiment_prompt(train_code, results_history, best_bpb, crash_info=None):
    """Build the prompt for the LLM to propose an experiment."""

    hyper_lines = extract_hyperparams_as_lines(train_code)

    prompt = f"""You are an autonomous ML researcher optimizing a GPT training script.

GOAL: Lower val_bpb (bits per byte on validation set). Current best: {best_bpb:.6f}

AVAILABLE HYPERPARAMETERS (current values — these are the ONLY variables you can change):
{hyper_lines}

CONSTRAINTS:
- TOTAL_BATCH_SIZE must be divisible by (DEVICE_BATCH_SIZE * 2048)
- Do NOT change FINAL_EVAL_BATCH_SIZE
- Training runs for a fixed 5-minute budget on Apple Silicon MPS

EXPERIMENT HISTORY:
{results_history}

{"LAST CRASH:" + chr(10) + crash_info if crash_info else ""}

INSTRUCTIONS:
Pick ONE hyperparameter from the list above and propose a new value to improve val_bpb.
Give one sentence of reasoning.

OUTPUT: One line only, exactly like this example:
MATRIX_LR = 0.02
"""
    return prompt


def parse_variable_change(response):
    """Parse a simple VARIABLE = value assignment from LLM response."""
    for line in response.strip().split("\n"):
        line = re.sub(r'^[`*#\s]+|[`*#\s]+$', '', line).strip()
        m = re.match(r'^([A-Z][A-Z_0-9]*)\s*=\s*(.+)$', line)
        if m:
            var_name = m.group(1)
            new_value = m.group(2).strip().rstrip(',')
            return var_name, new_value
    return None, None


def apply_variable_change(code, var_name, new_value):
    """Replace a top-level variable assignment in train.py, preserving comments."""
    lines = code.split("\n")
    for i, line in enumerate(lines):
        if re.match(rf'^{re.escape(var_name)}\s*=\s*', line):
            comment_match = re.search(r'\s{2,}#.*$', line)
            comment = comment_match.group(0) if comment_match else ''
            lines[i] = f'{var_name} = {new_value}{comment}'
            return "\n".join(lines), True
    return code, False


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Autoresearch Local LLM Agent")
    print(f"Model: {MODEL}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    init_results()
    consecutive_crashes = 0
    experiment_num = 0

    # Check if baseline exists
    results = get_results_history()
    if "baseline" not in results.lower():
        print("\n--- Experiment 0: Baseline ---")
        base_commit = get_current_commit()
        val_bpb, peak_vram = run_experiment()
        if val_bpb is not None:
            memory_gb = peak_vram / 1024 if peak_vram else 0
            log_result(base_commit, val_bpb, memory_gb, "keep", "baseline")
            print(f"  Baseline val_bpb: {val_bpb:.6f}")
        else:
            crash_info = get_crash_info()
            print(f"  Baseline run failed! Check run.log")
            print(f"  Last lines: {crash_info[-500:]}")
            log_result(base_commit, 0.0, 0.0, "crash", "baseline failed")
            print("  Continuing anyway to let LLM try fixes...")
        experiment_num = 1

    # Main loop — runs indefinitely until manually stopped
    while True:
        print(f"\n{'=' * 60}")
        print(f"--- Experiment {experiment_num} ---")
        print(f"Time: {datetime.now().isoformat()}")

        best_bpb = get_best_bpb()
        base_commit = get_current_commit()
        train_code = read_train_py()
        results_history = get_results_history()

        crash_context = None
        if consecutive_crashes > 0:
            crash_context = get_crash_info()

        # Ask LLM for a modification
        print("  Querying LLM for experiment proposal...")
        prompt = build_experiment_prompt(train_code, results_history, best_bpb, crash_context)
        response = query_llm(prompt, max_tokens=4096)

        if not response:
            print("  LLM returned empty response, waiting 30s...")
            time.sleep(30)
            experiment_num += 1
            continue

        # Parse variable change
        var_name, new_value = parse_variable_change(response)
        if not var_name:
            print("  Could not parse VARIABLE = value from response, skipping")
            preview = response[:300].replace("\n", "\n    ")
            print(f"    Response preview:\n    {preview}")
            experiment_num += 1
            consecutive_crashes += 1
            if consecutive_crashes >= MAX_CONSECUTIVE_CRASHES:
                print(f"  {MAX_CONSECUTIVE_CRASHES} consecutive failures, resetting...")
                consecutive_crashes = 0
            continue

        # Apply variable change
        modified_code, success = apply_variable_change(train_code, var_name, new_value)
        if not success:
            print(f"  Variable '{var_name}' not found in train.py, skipping")
            experiment_num += 1
            consecutive_crashes += 1
            if consecutive_crashes >= MAX_CONSECUTIVE_CRASHES:
                print(f"  {MAX_CONSECUTIVE_CRASHES} consecutive failures, resetting...")
                consecutive_crashes = 0
            continue

        # Validate syntax of modified code
        valid, error = validate_syntax(modified_code)
        if not valid:
            print(f"  Syntax error after applying changes: {error}")
            experiment_num += 1
            consecutive_crashes += 1
            if consecutive_crashes >= MAX_CONSECUTIVE_CRASHES:
                print(f"  {MAX_CONSECUTIVE_CRASHES} consecutive failures, resetting...")
                consecutive_crashes = 0
            continue

        description = f"{var_name} = {new_value}"

        # Apply and commit
        write_train_py(modified_code)
        print(f"  Applied: {description}")
        try:
            commit_hash = git_commit(f"exp{experiment_num}: {description}")
        except Exception as e:
            print(f"  Git commit failed: {e}")
            git_reset_hard(base_commit)
            experiment_num += 1
            continue

        # Run experiment
        val_bpb, peak_vram = run_experiment()

        if val_bpb is None:
            crash_info = get_crash_info()
            print(f"  CRASH: {crash_info[-200:]}")
            log_result(commit_hash, 0.0, 0.0, "crash", description)
            git_reset_hard(base_commit)
            consecutive_crashes += 1

            if consecutive_crashes >= MAX_CONSECUTIVE_CRASHES:
                print(f"  {MAX_CONSECUTIVE_CRASHES} consecutive crashes, resetting state...")
                consecutive_crashes = 0
        else:
            consecutive_crashes = 0
            memory_gb = peak_vram / 1024 if peak_vram else 0

            if val_bpb < best_bpb:
                log_result(commit_hash, val_bpb, memory_gb, "keep", description)
                print(f"  KEEP: {val_bpb:.6f} < {best_bpb:.6f} (improved by {best_bpb - val_bpb:.6f})")
            else:
                log_result(commit_hash, val_bpb, memory_gb, "discard", description)
                print(f"  DISCARD: {val_bpb:.6f} >= {best_bpb:.6f}")
                git_reset_hard(base_commit)

        experiment_num += 1
        print(f"  Best val_bpb so far: {get_best_bpb():.6f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAgent stopped by user.")
        print(f"Final results:\n{get_results_history()}")
        sys.exit(0)
