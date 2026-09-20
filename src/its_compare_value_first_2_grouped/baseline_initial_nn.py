import copy
import numpy as np
import torch
import torch.nn.functional as F

from its_nn_value_first_grouped.config import add_args as add_clean4_args
from its_nn_value_first_grouped.model import Net
from its_nn_value_first_grouped.run_scaf_server import set_global_seed


def _make_nn_args(base_args):
    import argparse
    parser = argparse.ArgumentParser()
    add_clean4_args(parser)
    args = parser.parse_args([])

    fields = [
        "n_features", "n_actions", "n_hidden_layers", "hidden1", "hidden2",
        "dropout", "use_baseline", "train_epochs", "train_lr", "train_wd",
        "patience", "eval_every", "global_seed", "deterministic"
    ]
    for key in fields:
        if hasattr(base_args, key):
            setattr(args, key, getattr(base_args, key))
    return args


class InitialNetworkPolicy:
    def __init__(self, args):
        self.args = _make_nn_args(copy.deepcopy(args))
        self.model = None
        self.best_val_mse_ = None

    def fit(self, X, A, Y, X_val=None, A_val=None, Y_val=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_global_seed(
            int(getattr(self.args, "global_seed", 123)),
            deterministic=bool(getattr(self.args, "deterministic", False)),
        )

        self.model = Net(
            n_features=self.args.n_features,
            n_treatments=self.args.n_actions,
            hidden1=self.args.hidden1,
            hidden2=self.args.hidden2,
            dropout=self.args.dropout,
            use_baseline=self.args.use_baseline,
            n_hidden_layers=self.args.n_hidden_layers,
        ).to(device)

        Xtr = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
        Atr = torch.tensor(np.asarray(A, dtype=np.int64), device=device)
        Ytr = torch.tensor(np.asarray(Y, dtype=np.float32), device=device).view(-1)

        Xva = torch.tensor(np.asarray(X_val, dtype=np.float32), device=device)
        Ava = torch.tensor(np.asarray(A_val, dtype=np.int64), device=device)
        Yva = torch.tensor(np.asarray(Y_val, dtype=np.float32), device=device).view(-1)

        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.args.train_lr),
            weight_decay=float(self.args.train_wd),
        )

        best_state = None
        best_val = float("inf")
        bad = 0

        epochs = int(self.args.train_epochs)
        patience = int(self.args.patience)
        eval_every = max(1, int(self.args.eval_every))

        for ep in range(1, epochs + 1):
            self.model.train()
            opt.zero_grad()
            pred = self.model(Xtr, Atr).view(-1)
            loss = F.mse_loss(pred, Ytr)
            loss.backward()
            opt.step()

            if ep % eval_every == 0 or ep == epochs:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(Xva, Ava).view(-1)
                    val_mse = float(F.mse_loss(val_pred, Yva).item())

                if val_mse < best_val - 1e-6:
                    best_val = val_mse
                    best_state = copy.deepcopy(self.model.state_dict())
                    bad = 0
                else:
                    bad += 1

                if bad >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.best_val_mse_ = float(best_val)
        return self

    def predict_mu(self, X):
        X_np = np.asarray(X, dtype=np.float32)
        device = next(self.model.parameters()).device
        X_t = torch.tensor(X_np, dtype=torch.float32, device=device)

        self.model.eval()
        with torch.no_grad():
            mu = self.model.predict_all_treatments(X_t).detach().cpu().numpy()
        return mu

    def predict_treatment(self, X):
        return self.predict_mu(X).argmax(axis=1)

    def get_diagnostics(self):
        return {
            "initial_nn_best_val_mse": getattr(self, "best_val_mse_", np.nan),
        }
