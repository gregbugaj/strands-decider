# Installing and using Strands Decider with Astral uv

This guide covers the local source checkout, inference, development, and the
recommended migration to a locked `uv` project. The migration section is a proposal;
this checkout currently has no committed `uv.lock` or CPU/CUDA extras.

## Install from this checkout

Use Python 3.12 and a separate environment for this project. On the machine where
this guide was written, Python 3.12 and `uv` were already installed. Check with:

```bash
python3 --version
uv --version
```

If `uv` is missing, follow the [Astral installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

```bash
cd /home/gbugaj/dev/marieai/strands-decider

uv venv --python 3.12
source .venv/bin/activate

uv pip install --torch-backend=auto -e '.[dev]'
strands-decider --help
```

`-e` installs the local project in editable mode, so source changes are available
without reinstalling. The `dev` extra adds development tools; for inference only,
replace `'.[dev]'` with `.`.

`--torch-backend=auto` selects the PyTorch backend based on the installed GPU
driver, falling back to CPU when no supported GPU is detected. To install CPU-only
dependencies explicitly, use this instead of the install command above:

```bash
uv pip install --torch-backend=cpu -e '.[dev]'
```

See [Astral's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

Each time you return to the project, activate its environment:

```bash
cd /home/gbugaj/dev/marieai/strands-decider
source .venv/bin/activate
```

Use the activated environment's commands for this workflow. `uv run` and `uv sync`
manage a project lockfile and may resolve or change the environment; the migration
below establishes that workflow explicitly.

## Ask the model

The first inference run downloads the model from Hugging Face and needs network
access. Installing the Python package does not download the model weights.

Choose among options:

```bash
strands-decider ask StrandsAgents/strands-decider-2B-hobson-v19 \
  --state "My payouts have been failing for three days." \
  --choice "Which team should handle this?=billing,sales,retail"
```

Ask a yes/no question:

```bash
strands-decider ask StrandsAgents/strands-decider-2B-hobson-v19 \
  --state "My payouts have been failing for three days." \
  --noul "Does this convey urgency?"
```

You can combine `--choice`, `--noul`, and `--score` in one command. Add `--json`
for the raw API response. The CLI automatically selects CUDA, then MPS, then CPU;
use `--device cuda`, `--device mps`, or `--device cpu` to select explicitly.

## Run the HTTP server

```bash
strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 \
  --host 127.0.0.1 --port 8000
```

From another terminal:

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "state": "My payouts have been failing for three days.",
    "questions": {
      "is_urgent": {
        "type": "noul",
        "instructions": "Does this convey urgency?"
      }
    }
  }'
```

Stop the server with `Ctrl+C`.

## Development checks

With the `dev` extra installed and the environment activated:

```bash
pytest -q
ruff check .
mypy ./src
```

GPU tests skip when CUDA is unavailable. Distributed tests are excluded from the
default suite; run `pytest -q -m distributed` separately when needed.

For contributor instructions and optional Git hooks, see [CONTRIBUTING.md](CONTRIBUTING.md#development).

## Training setup differs from inference

The `train` extra adds `datasets` for corpus building, but does not install the
complete documented GPU training environment. The Qwen3.5 recipe documents
Torch 2.7.1/CUDA 12.6, Transformers 5.17.0, PEFT 0.21.0, and
`flash-linear-attention` on Linux or WSL2.

Follow [training/README.md](training/README.md#setup) before training. An automatic
backend install may select newer versions than that recipe uses.

## Recommended uv migration

Adopt Astral `uv` for reproducible local and CI installs:

1. Select Python 3.12 for local development and commit a generated `uv.lock`.
2. Add mutually exclusive CPU/CUDA extras with explicit PyTorch indexes.
3. Put development tools in a `dev` dependency group.
4. Update installation documentation and CI to use `uv sync --locked` and `uv run`.
5. Retain setuptools and setuptools-scm for builds and versioning.
6. Preserve the documented training stack initially; validate CUDA upgrades separately.

After those changes, a CPU development setup could use:

```bash
# Proposed commands: require the migration above first.
uv sync --locked --extra cpu --group dev
uv run --locked --extra cpu --group dev strands-decider --help
uv run --locked --extra cpu --group dev pytest -q
```

See [uv project management](https://docs.astral.sh/uv/guides/projects/) for lockfiles
and [PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/)
for configuring accelerator extras.

### Keep the environment separate from Marie-AI

At the time of inspection, Strands Decider required Transformers >=5.15 and
PEFT >=0.21. Marie-AI's CUDA extra pinned Transformers 5.13.0 and PEFT 0.19.1.
Those requirements conflict. Share the `uv` workflow, and keep separate project
environments and lockfiles.

## Validation recorded on 2026-10-02

A CPU installation dry run with Python 3.12.3 and uv 0.11.28 successfully resolved
80 packages. That check did not install dependencies, execute the tests, download
model weights, or validate inference or GPU training. Dependency versions may
change until a lockfile is committed.
