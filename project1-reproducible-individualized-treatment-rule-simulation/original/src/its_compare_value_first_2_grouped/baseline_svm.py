import itertools
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class SVMPolicy:
    def __init__(
        self,
        n_actions,
        param_grid=None,
        refit_on_trainval=True,
        min_samples_per_action=5,
    ):
        self.n_actions = int(n_actions)
        self.refit_on_trainval = bool(refit_on_trainval)
        self.min_samples_per_action = int(min_samples_per_action)
        self.param_grid = param_grid or {
            "C": [1.0, 10.0],
            "epsilon": [0.1, 0.2],
            "gamma": ["scale"],
            "kernel": ["rbf"],
        }

    def _grid_candidates(self):
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))

    def _make_model(self, svm_kwargs):
        return make_pipeline(StandardScaler(), SVR(**svm_kwargs))

    def _fit_models(self, X, A, Y, svm_kwargs):
        models = []
        action_counts = {}
        action_means = {}
        for a in range(self.n_actions):
            idx = (A == a)
            n_a = int(idx.sum())
            action_counts[a] = n_a
            if n_a == 0:
                action_means[a] = float(np.mean(Y))
                models.append(None)
                continue
            action_means[a] = float(np.mean(Y[idx]))
            if n_a < self.min_samples_per_action:
                models.append(None)
                continue
            model = self._make_model(svm_kwargs)
            model.fit(X[idx], Y[idx])
            models.append(model)
        return models, action_counts, action_means

    def _predict_single_action(self, model, fallback_mean, X):
        if model is None:
            return np.full(X.shape[0], fallback_mean, dtype=float)
        return model.predict(X)

    def _observed_mse(self, models, action_means, X, A, Y):
        y_hat = np.zeros(len(A), dtype=float)
        for a in range(self.n_actions):
            idx = (A == a)
            if np.any(idx):
                y_hat[idx] = self._predict_single_action(models[a], action_means[a], X[idx])
        return float(np.mean((Y - y_hat) ** 2))

    def fit(self, X, A, Y, X_val=None, A_val=None, Y_val=None):
        X = np.asarray(X)
        A = np.asarray(A)
        Y = np.asarray(Y)
        if X_val is None or A_val is None or Y_val is None:
            params = next(self._grid_candidates())
            self.models, self.action_counts_, self.action_means_ = self._fit_models(X, A, Y, params)
            self.best_params_ = params
            self.best_val_mse_ = None
            return self

        best = None
        best_params = None
        best_val = np.inf
        best_counts = None
        best_means = None
        for svm_kwargs in self._grid_candidates():
            models, counts, means = self._fit_models(X, A, Y, svm_kwargs)
            val_mse = self._observed_mse(models, means, np.asarray(X_val), np.asarray(A_val), np.asarray(Y_val))
            if val_mse < best_val:
                best_val = val_mse
                best = models
                best_params = svm_kwargs
                best_counts = counts
                best_means = means
        if self.refit_on_trainval:
            X_refit = np.concatenate([X, X_val], axis=0)
            A_refit = np.concatenate([A, A_val], axis=0)
            Y_refit = np.concatenate([Y, Y_val], axis=0)
            best, best_counts, best_means = self._fit_models(X_refit, A_refit, Y_refit, best_params)
        self.models = best
        self.action_counts_ = best_counts
        self.action_means_ = best_means
        self.best_params_ = best_params
        self.best_val_mse_ = float(best_val)
        return self

    def predict_mu(self, X):
        X = np.asarray(X)
        preds = []
        for a in range(self.n_actions):
            preds.append(self._predict_single_action(self.models[a], self.action_means_[a], X))
        return np.column_stack(preds)

    def predict_treatment(self, X):
        return self.predict_mu(X).argmax(axis=1)

    def get_diagnostics(self):
        return {
            "svm_best_val_mse": getattr(self, "best_val_mse_", np.nan),
            "svm_best_params": str(getattr(self, "best_params_", None)),
        }
