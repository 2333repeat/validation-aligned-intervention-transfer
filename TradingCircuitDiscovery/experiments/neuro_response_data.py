"""独立状态构建与冻结模型的真实集合干预。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def dated_states(frame, start, end, *, lookback=20):
    """沿用现有七特征、窗口内样本标准差；预处理不跨指定数据段。"""
    data = frame.copy()
    data['Date'] = pd.to_datetime(data.Date)
    data = data.loc[(data.Date >= start) & (data.Date <= end) & (data.Volume != 0)].copy()
    data = data.sort_values('Date').drop_duplicates('Date').reset_index(drop=True)
    for field in ['Open', 'High', 'Low', 'Close', 'Volume']:
        data[field] = pd.to_numeric(data[field], errors='raise').astype(float)
    data['MA_5'] = data.Close.rolling(5, min_periods=1).mean()
    data['MA_10'] = data.Close.rolling(10, min_periods=1).mean()
    features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_10']]
    rows, indices = [], []
    # 与历史环境相同，保留末尾 lookback+1 行以保持回顾性口径。
    for t in range(lookback - 1, len(data) - lookback - 1):
        window = features.iloc[t + 1 - lookback:t + 1]
        state = ((window - window.mean()) / (window.std() + 1e-8)).to_numpy().ravel()
        if not np.isfinite(state).all():
            raise ValueError('nonfinite market state')
        rows.append(state)
        indices.append(t)
    if not rows:
        raise ValueError('insufficient dated observations')
    return np.asarray(rows, dtype=np.float32), data.iloc[indices].Date.dt.strftime('%Y-%m-%d').to_numpy(), data, indices


class JointIntervention:
    """仅缓存冻结层之前的前向；之后的非线性仍为真实联合前向。"""
    def __init__(self, adapter, states, layer, discovery_means):
        component, name = layer.split('.')
        self.network = getattr(adapter.agent, component)
        self.network.eval()
        self.layer = layer
        self.index = int(name.removeprefix('fc'))
        self.depth = 3 if hasattr(self.network, 'fc3') else 2
        if self.index >= self.depth:
            raise ValueError('only hidden linear layers may be intervened on')
        self.device = next(self.network.parameters()).device
        self.means = torch.as_tensor(discovery_means, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            x = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            for index in range(1, self.index + 1):
                x = getattr(self.network, f'fc{index}')(x)
                if index < self.index:
                    x = torch.relu(x)
            self.hidden = x
            self.baseline_logits = self._suffix(x).detach()
            self.baseline = torch.softmax(self.baseline_logits, dim=-1).cpu().numpy()
        self.cache = {(): self.baseline}

    def _suffix(self, hidden):
        values = torch.relu(hidden)
        for index in range(self.index + 1, self.depth + 1):
            values = getattr(self.network, f'fc{index}')(values)
            if index < self.depth:
                values = torch.relu(values)
        return values

    def probabilities(self, indices, *, temperature=1.0):
        key = tuple(sorted(set(int(i) for i in indices)))
        if temperature == 1.0 and key in self.cache:
            return self.cache[key]
        with torch.no_grad():
            hidden = self.hidden.clone()
            hidden[:, list(key)] = self.means[list(key)]
            output = torch.softmax(self._suffix(hidden) / temperature, dim=-1).cpu().numpy()
        if temperature == 1.0:
            self.cache[key] = output
        return output

    def response(self, indices):
        return self.probabilities(indices) - self.baseline

    def base_at_temperature(self, temperature):
        return torch.softmax(self.baseline_logits / temperature, dim=-1).cpu().numpy()

    def raw_logits(self, indices):
        key = tuple(sorted(set(int(i) for i in indices)))
        with torch.no_grad():
            hidden = self.hidden.clone()
            hidden[:, list(key)] = self.means[list(key)]
            return self._suffix(hidden).cpu().numpy()
