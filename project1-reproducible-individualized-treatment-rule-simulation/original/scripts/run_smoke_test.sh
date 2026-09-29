#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
SMOKE_OUT="${NSCF_SMOKE_OUT:-demo_outputs}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SMOKE_OUT}/.mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

python -m its_compare_value_first_2_grouped.sim_compare_formal \
  --out_base "${SMOKE_OUT}" \
  --run_name smoke_grouped_nonlinear \
  --scenarios grouped_nonlinear_hard \
  --groups "0,1,2,3,4|5,6,7,8,9" \
  --n_reps 1 \
  --seed0 123 \
  --n_train 80 \
  --n_val 40 \
  --n_test 200 \
  --n_features 10 \
  --n_actions 10 \
  --train_epochs 5 \
  --alt_cycles 1 \
  --admm_epochs 5 \
  --penalty l1 \
  --num_lambda 3 \
  --refit_select_epochs 5 \
  --rf_n_jobs 1

echo "Smoke test completed. Outputs were written under ${SMOKE_OUT}/."
