import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    Regression with categorical treatment A:
        h = H(x) in R^d
        gamma[a] in R^d  (to be fused across treatments)
        y_hat = m(x) + <h, gamma[a]>

    Flexible backbone:
        - n_hidden_layers = 1: x -> hidden_dim -> h
        - n_hidden_layers = 2: x -> hidden1 -> hidden2 -> h
    """
    def __init__(
        self,
        n_features,
        n_treatments,
        hidden1=32,
        hidden2=16,
        dropout=0.0,
        use_baseline=True,
        n_hidden_layers=1,
    ):
        super().__init__()

        self.n_treatments = int(n_treatments)
        self.use_baseline = bool(use_baseline)
        self.n_hidden_layers = int(n_hidden_layers)

        if self.n_hidden_layers not in [1, 2]:
            raise ValueError(f"n_hidden_layers must be 1 or 2, got {self.n_hidden_layers}")

        # -------- backbone --------
        if self.n_hidden_layers == 1:
            # x -> hidden2
            self.fc1 = nn.Linear(n_features, hidden2)
            self.fc2 = None
            self.feature_dim = int(hidden2)

        else:
            # x -> hidden1 -> hidden2
            self.fc1 = nn.Linear(n_features, hidden1)
            self.fc2 = nn.Linear(hidden1, hidden2)
            self.feature_dim = int(hidden2)

        self.dropout = nn.Dropout(p=dropout)

        # treatment-specific parameter to be fused: gamma[a] in R^feature_dim
        self.gamma = nn.Embedding(self.n_treatments, self.feature_dim)

        # optional baseline m(x)
        if self.use_baseline:
            self.m_head = nn.Linear(self.feature_dim, 1, bias=True)
        else:
            self.m_head = None

        # initialize gamma small (helps stability)
        nn.init.normal_(self.gamma.weight, mean=0.0, std=0.02)

    def extract_features(self, x):
        """
        x: (n, p)
        return h: (n, d)
        """
        h = self.fc1(x)
        h = F.relu(h)
        h = self.dropout(h)

        if self.n_hidden_layers == 2:
            h = self.fc2(h)
            h = F.relu(h)
            h = self.dropout(h)

        return h

    def forward(self, x, a):
        """
        x: (n, p)
        a: (n,) long tensor with values in {0,...,K-1}
        returns y_hat: (n,) float tensor
        """
        h = self.extract_features(x)              # (n, d)
        g = self.gamma(a.long())                  # (n, d)

        # interaction term <h, g>
        inter = torch.sum(h * g, dim=1)           # (n,)

        if self.m_head is not None:
            base = self.m_head(h).squeeze(1)      # (n,)
            y_hat = base + inter
        else:
            y_hat = inter

        return y_hat

    def predict_from_HA(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        H: (n, d) already-extracted features
        A: (n,)
        """
        g = self.gamma(A.long())                  # (n, d)
        inter = torch.sum(H * g, dim=1)          # (n,)

        if self.m_head is not None:
            base = self.m_head(H).squeeze(1)     # (n,)
            return base + inter
        return inter

    def gamma_matrix(self):
        """Return gamma as a (K, d) Parameter-like tensor for ADMM."""
        return self.gamma.weight

    def predict_all_treatments(self, x):
        """
        x: (n, p)
        return:
            y_all: (n, K), where [i, a] is predicted outcome for sample i under treatment a
        """
        h = self.extract_features(x)  # (n, d)
        n = x.size(0)
        K = self.n_treatments

        outs = []
        for a in range(K):
            A = torch.full((n,), a, dtype=torch.long, device=x.device)
            y_a = self.predict_from_HA(h, A)  # (n,)
            outs.append(y_a)

        return torch.stack(outs, dim=1)  # (n, K)
