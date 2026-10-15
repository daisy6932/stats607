PYTHON ?= python3
CONFIG ?= config/study.json
OUTPUT ?= results

.PHONY: all simulate analyze figures clean test smoke

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
