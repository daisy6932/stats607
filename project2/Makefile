PYTHON_BIN ?= python3
VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
REQUIREMENTS ?= requirements.txt
CONFIG ?= config/study.json
OUTPUT ?= results

.PHONY: setup all simulate analyze figures clean test smoke

setup:
	"$(PYTHON_BIN)" -c 'import sys; sys.exit("Use Python 3.13 or newer: make setup PYTHON_BIN=python3.13") if sys.version_info < (3, 13) else None'
	"$(PYTHON_BIN)" -m venv "$(VENV)"
	"$(VENV)/bin/python" -m pip install --only-binary=:all: -r "$(REQUIREMENTS)"
	"$(VENV)/bin/python" -m pip check

# Run targets sequentially even with make -j; analysis reads saved raw output.
all: figures

simulate:
	$(PYTHON) -m src.simulation --config "$(CONFIG)" --output "$(OUTPUT)"

analyze: simulate
	$(PYTHON) -m src.analyze --output "$(OUTPUT)"

figures: analyze
	$(PYTHON) -m src.figures --output "$(OUTPUT)"

test:
	$(PYTHON) -m pytest

smoke:
	$(MAKE) all CONFIG=config/smoke.json OUTPUT=tmp/smoke

# Deliberately restricted to the documented default generated files. Preserves reference data.
clean:
	$(PYTHON) -c 'from pathlib import Path; import shutil; p=Path("results"); [shutil.rmtree(p/d) for d in ("raw", "figures") if (p/d).exists()]; [(p/f).unlink(missing_ok=True) for f in ("summary.csv", "diagnostics.csv", "paired_power.csv", "design_matrix.csv", "table2_comparison.csv", "pi0_expectation.csv")]; shutil.rmtree("tmp/smoke", ignore_errors=True)'
