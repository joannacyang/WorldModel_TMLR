# LLM-Based World Models Can Make Decisions Solely, But Rigorous Evaluations are Needed

This repository contains the code and experiments for the paper **"LLM-Based World Models Can Make Decisions Solely, But Rigorous Evaluations are Needed"**. It implements an LLM-based world model for text-based simulation games, evaluating whether such models can (1) predict next states from actions, (2) propose useful actions, and (3) plan to complete tasks using only the model (no environment) in the final steps.

## Overview

The world model is implemented in `world/model.py`. It uses an LLM (GPT-4o or GPT-4o-mini) with:

- **Human-written rules** for object properties, actions, and scoring (`world/rules/human_written_rules/`)
- **Structured state representation**: game state as JSON; the model predicts `modified` / `removed` object states and score after each action
- **Few-shot examples** of state transitions (action-induced and time-induced) from `world/examples.json` and `data/dynamic_states.json`

The benchmark consists of **30 text games** in `games/` (e.g. `lit-lightbulb`, `cooking`, `wash-clothes`). Each game exposes a `TextGame` class with `step()`, `getTaskDescription()`, `generatePossibleActions()`, and `get_demo_actions()` (gold action sequence for evaluation).

## Setup

### 1. API key

Place your OpenAI API key in:

```text
api_keys/openai_api_key.txt
```

(One line, no extra spaces or newlines.)

### 2. Python environment

From the project root:

```bash
pip install -r requirements.txt
```

- **Python 3.8+** recommended.
- The code uses `openai.OpenAI()` by default; for Azure, the code supports `openai.api_type == "azure"` (see `world/utils.py`).

### 3. Data and rules

Required paths:

| Path | Description |
|------|-------------|
| `world/examples.json` | Few-shot state-transition examples for the world model |
| `data/dynamic_states.json` | Fine-grained state-change info (action vs. time) used in prompts |
| `world/rules/human_written_rules/object_rules.json` | Object property descriptions per game |
| `world/rules/human_written_rules/action_rules.json` | Action descriptions per game |
| `world/rules/human_written_rules/score_rules.json` | Score function descriptions per game |

Results are written under `results/results_data/`; create that directory if needed.

## Running the experiments

All three entry points use the game list from `world/__init__.py` (`avail_games["games"]`). **Run all commands from the project root** (required for `./api_keys/`, `./data/`, etc.). All three scripts load the API key from `api_keys/openai_api_key.txt` (via `world/utils.py`).

### 1. Policy verification

Evaluates how well the world model’s **state (and score) predictions** match the environment when following demo actions. For the last fraction of the trajectory, the model is used without the environment (`use_env=False`); correctness is measured by comparing final predicted state/score to the real one.

```bash
python main_policy_verification.py [options]
```

Useful options:

- `--model` — `gpt-4o` or `gpt-4o-mini` (default: `gpt-4o-mini`)
- `--last_steps_to_verify` — Fraction of demo steps where the model is used without the env (default: `1.0`)
- `--max_try` — Number of runs per game (default: `30`)

Results: `results/results_data/results_verifying_<last_steps_to_verify>_<model>.json`

### 2. Action proposal

Evaluates whether the world model’s **top-k suggested actions** contain the gold demo action. Proposed actions are matched to the environment’s available actions via embedding similarity (`world/utils.py`: `query_actions` with `text-embedding-3-small`).

```bash
python main_action_proposal.py [options]
```

Useful options:

- `--model` — e.g. `gpt-4o` (default: `gpt-4o`)
- `--topk` — Number of proposed actions to consider (default: `10`)
- `--max_try` — Number of runs per game (default: `30`)

Results: `results/results_data/results_action_proposal_<topk>_<model>.json`

### 3. Policy planning

Runs the **model-only decision** experiment: execute gold demo actions for the first part of the trajectory, then switch to the world model for the rest. The model proposes a single next action (`get_action_proposals(k=1)`), predicts the next state, and repeats until `gameOver` or max steps. Success is whether the game is won.

```bash
python main_policy_planning.py [options]
```

Useful options:

- `--model` — e.g. `gpt-4o-mini` (default: `gpt-4o-mini`)
- `--last_steps_to_find` — Fraction of demo length to “leave” for planning (default: `1.0`)
- `--max_try` — Number of runs per game (default: `50`)

Results: result saving is commented out in the script; success is printed per run. To save to JSON, uncomment the `results_folder` / `json.dump` block in `main_policy_planning.py`.

## Project structure

```text
.
├── LICENSE
├── README.md
├── requirements.txt
├── main_policy_verification.py   # State/score prediction evaluation
├── main_action_proposal.py       # Action proposal (top-k) evaluation
├── main_policy_planning.py       # Model-only planning evaluation
├── api_keys/
│   └── openai_api_key.txt        # Your OpenAI API key (create this)
├── data/
│   └── dynamic_states.json       # State-change metadata for prompts
├── games/                        # 30 text game modules (e.g. lit-lightbulb.py, cooking.py)
├── world/
│   ├── __init__.py               # avail_games, game lists by difficulty
│   ├── model.py                  # WorldModel: step, get_predicted_state, get_action_proposals, planning
│   ├── make_state.py            # State serialization and partial state recovery
│   ├── utils.py                  # OpenAI client, stream_llm_gpt, query_actions (embeddings)
│   ├── examples.json             # Few-shot examples for the world model
│   └── rules/
│       ├── human_written_rules/  # object_rules.json, action_rules.json, score_rules.json
│       └── ...
├── world_model/
│   └── model.py                 # Alternate world model (LLM-generated rules)
└── results/
    └── results_data/             # Output JSON result files
```


