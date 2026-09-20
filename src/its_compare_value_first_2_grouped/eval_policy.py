import numpy as np


def evaluate_policy(mu_true, a_hat, a_opt):
    mu_true = np.asarray(mu_true)
    a_hat = np.asarray(a_hat)
    a_opt = np.asarray(a_opt)

    value = mu_true[np.arange(len(a_hat)), a_hat].mean()
    oracle_value = mu_true[np.arange(len(a_opt)), a_opt].mean()
    regret = oracle_value - value
    row_max = mu_true.max(axis=1)
    chosen_value = mu_true[np.arange(len(a_hat)), a_hat]
    acc_opt = np.isclose(chosen_value, row_max, atol=1e-8, rtol=1e-8).mean()

    return {
        "value": float(value),
        "oracle_value": float(oracle_value),
        "regret": float(regret),
        "acc_opt": float(acc_opt),
    }
