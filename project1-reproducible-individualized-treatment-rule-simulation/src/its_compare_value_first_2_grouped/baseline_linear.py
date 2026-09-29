import itertools
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet


class LinearPolicy:
    def __init__(
        self,
        n_actions,
        model_type="ridge",
        param_grid=None,
        random_state=0,
        refit_on_trainval=True,
        min_samples_per_action=5,
    ):
        self.n_actions = int(n_actions)
        self.model_type = str(model_type)
        self.random_state = int(random_state)
        self.refit_on_trainval = bool(refit_on_trainval)
        self.min_samples_per_action = int(min_samples_per_action)

        if param_grid is not None:
            self.param_grid = param_grid
        elif self.model_type == "ridge":
            self.param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
        elif self.model_type == "lasso":
            self.param_grid = {"alpha": [0.001, 0.01, 0.1, 1.0]}
        elif self.model_type == "elasticnet":
            self.param_grid = {
                "alpha": [0.001, 0.01, 0.1, 1.0],
                "l1_ratio": [0.2, 0.5, 0.8],
            }
        else:
            raise ValueError(f"Unknown model_type={self.model_type}")

    def _grid_candidates(self):
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))

    def _make_model(self, kwargs):
        if self.model_type == "ridge":
            model = Ridge(**kwargs)
        elif self.model_type == "lasso":
            model = Lasso(**kwargs, max_iter=20000, random_state=self.random_state)
        elif self.model_type == "elasticnet":
            model = ElasticNet(**kwargs, max_iter=20000, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown model_type={self.model_type}")
        return make_pipeline(StandardScaler(), model)

    def _fit_models(self, X, A, Y, kwargs):
        models = []
        action_counts = {}
        action_means = {}
        global_mean = float(np.mean(Y))

        for a in range(self.n_actions):
            idx = (A == a)
            n_a = int(idx.sum())
            action_counts[a] = n_a

            if n_a == 0:
                action_means[a] = global_mean
                models.append(None)
                continue

            action_means[a] = float(np.mean(Y[idx]))

            if n_a < self.min_samples_per_action:
                models.append(None)
                continue

            model = self._make_model(kwargs)
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

        X_val = np.asarray(X_val)
        A_val = np.asarray(A_val)
        Y_val = np.asarray(Y_val)

        best_models = None
        best_params = None
        best_val = np.inf
        best_counts = None
        best_means = None

        for params in self._grid_candidates():
            models, counts, means = self._fit_models(X, A, Y, params)
            val_mse = self._observed_mse(models, means, X_val, A_val, Y_val)
            if val_mse < best_val:
                best_val = val_mse
                best_models = models
                best_params = params
                best_counts = counts
                best_means = means

        if self.refit_on_trainval:
            X_refit = np.concatenate([X, X_val], axis=0)
            A_refit = np.concatenate([A, A_val], axis=0)
            Y_refit = np.concatenate([Y, Y_val], axis=0)
            best_models, best_counts, best_means = self._fit_models(
                X_refit, A_refit, Y_refit, best_params
            )

        self.models = best_models
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
            f"{self.model_type}_best_val_mse": getattr(self, "best_val_mse_", np.nan),
            f"{self.model_type}_best_params": str(getattr(self, "best_params_", None)),
        }
