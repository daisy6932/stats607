PYTHON_BIN ?= python3
VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
REQUIREMENTS ?= requirements.txt
ENV_READY := $(VENV)/.dependencies-ready
CONFIG ?= config/study.json
OUTPUT ?= results

.PHONY: setup all simulate analyze figures clean test smoke

setup: $(ENV_READY)

$(ENV_READY): Makefile $(REQUIREMENTS)
	"$(PYTHON_BIN)" -c 'import sys; sys.exit("Use Python 3.11 or newer: make all PYTHON_BIN=python3.11") if sys.version_info < (3, 11) else None'
	"$(PYTHON_BIN)" -m venv "$(VENV)"
	"$(VENV)/bin/python" -m pip install -r "$(REQUIREMENTS)"
	"$(VENV)/bin/python" -m pip check
	touch "$@"

# Run targets sequentially even with make -j; analysis reads saved raw output.
all: figures

simulate: $(ENV_READY)
	$(PYTHON) -m src.simulation --config "$(CONFIG)" --output "$(OUTPUT)"

analyze: simulate
	$(PYTHON) -m src.analyze --output "$(OUTPUT)"

figures: analyze
	$(PYTHON) -m src.figures --output "$(OUTPUT)"

test: $(ENV_READY)
	$(PYTHON) -m pytest

smoke:
	$(MAKE) all CONFIG=config/smoke.json OUTPUT=tmp/smoke

clean:
	rm -rf results/raw results/figures tmp/smoke tmp/followup
	rm -f results/summary.csv results/diagnostics.csv results/paired_power.csv \
		results/design_matrix.csv results/table2_comparison.csv results/pi0_expectation.csv
