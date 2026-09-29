PYTHON ?= python
export PYTHONPATH := $(CURDIR)/src
export MPLBACKEND := Agg
export MPLCONFIGDIR := $(CURDIR)/artifacts/matplotlib
export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export CUDA_VISIBLE_DEVICES := -1
.PHONY: reproduce test compare
reproduce:
	$(PYTHON) src/pipeline.py
test:
	$(PYTHON) -m pytest -q
compare: reproduce
	$(PYTHON) src/compare_results.py
