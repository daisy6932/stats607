# STATS 607

## Project 1 — Reproducible Individualized Treatment Rule Simulation

The complete analysis is in [`project1-reproducible-individualized-treatment-rule-simulation/`](project1-reproducible-individualized-treatment-rule-simulation/).

See the [project README](project1-reproducible-individualized-treatment-rule-simulation/README.md) for the analysis, data source, environment, outputs, and validation.

### Setup and reproduction

From this repository root:

```bash
cd project1-reproducible-individualized-treatment-rule-simulation
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
make reproduce
make test
```

The reproduction command is `make reproduce`, run inside the project folder after environment setup. The `original` tag preserves the initial analysis before restructuring. Earlier commits place the project at the repository root; the latest structure groups it under its project name.
