"""
Clean implementation of the NN + fusion workflow.

This package keeps the synthetic data/model/ADMM components but rewrites the
selection logic around:
  - a fixed anchor pretrain baseline
  - cycle-start warm starts for grouped refits
  - explicit cycle-level model selection among grouped candidates
"""

