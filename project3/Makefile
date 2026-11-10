.DEFAULT_GOAL := all

PYTHON_BIN ?= python3
VENV ?= .venv
PYTHON ?= $(abspath $(VENV)/bin/python)
READY := $(VENV)/.dependencies-ready
CONFIG ?= config/study.json
OUTPUT ?= tmp/study
PERF_OUTPUT ?= tmp/performance
WORKERS ?= 2
STUDY_WORKERS ?= 1
TRIALS ?= 3

.PHONY: all setup simulate analyze figures test clean profile complexity benchmark parallel stability-check scaling performance baseline

setup: $(READY)

$(READY): Makefile requirements.txt
	"$(PYTHON_BIN)" -c 'import sys; sys.exit("Use Python 3.11+: make all PYTHON_BIN=python3.11") if sys.version_info < (3, 11) else None'
	"$(PYTHON_BIN)" -m venv "$(VENV)"
	"$(VENV)/bin/python" -m pip install -r requirements.txt
	"$(VENV)/bin/python" -m pip check
	touch "$@"

all: figures

simulate: $(READY)
	"$(PYTHON)" -m optimized --config "$(CONFIG)" --output "$(OUTPUT)" --workers "$(STUDY_WORKERS)"

analyze: simulate
	cd baseline && "$(PYTHON)" -m src.analyze --output "$(abspath $(OUTPUT))"

figures: analyze
	cd baseline && "$(PYTHON)" -m src.figures --output "$(abspath $(OUTPUT))"

test: $(READY)
	"$(PYTHON)" -m pytest

profile complexity benchmark scaling: $(READY)
	"$(PYTHON)" performance.py $@ --config "$(CONFIG)" --output "$(PERF_OUTPUT)" --trials "$(TRIALS)"

stability-check: $(READY)
	"$(PYTHON)" performance.py stability --config "$(CONFIG)" --output "$(PERF_OUTPUT)"

baseline: $(READY)
	"$(PYTHON)" performance.py baseline --config "$(CONFIG)" --output tmp/baseline --trials "$(TRIALS)"

parallel: $(READY)
	$(MAKE) all CONFIG="$(CONFIG)" OUTPUT=tmp/parallel-study STUDY_WORKERS="$(WORKERS)"

# Keep timed experiments sequential even if this target is invoked with make -j.
performance: $(READY)
	$(MAKE) -j1 benchmark complexity scaling
	"$(PYTHON)" plot_performance.py "$(PERF_OUTPUT)"

clean:
	rm -rf tmp/study tmp/parallel-study tmp/performance tmp/baseline
