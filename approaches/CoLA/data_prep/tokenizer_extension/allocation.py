from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass
class AllocationConfig:
    total_new_tokens: int = 128_000
    weight_ppw: float = 0.5
    weight_cpt: float = 0.5
    weight_unk: float = 0.0
    gamma: float = 2.0


def compute_allocation(metrics: pd.DataFrame, config: AllocationConfig) -> pd.DataFrame:
    df = metrics.copy()
    scaler = MinMaxScaler()
    df["ppw_n"] = scaler.fit_transform(df[["pieces_per_word"]])
    df["cpt_n"] = 1 - scaler.fit_transform(df[["chars_per_token"]])  # lower is better
    df["unk_n"] = scaler.fit_transform(df[["unknown_rate"]])

    # 1. linear combination
    df["I_lin"] = (
        config.weight_ppw * df["ppw_n"]
        + config.weight_cpt * df["cpt_n"]
        + config.weight_unk * df["unk_n"]
    )

    # 2. non-linear scaling to emphasize bad performing langs
    nonlinear = np.power(df["I_lin"].to_numpy(), config.gamma)
    df["inefficiency_score"] = MinMaxScaler().fit_transform(
        nonlinear.reshape(-1, 1)
    )

    # 3. allocate tokens proportionally to inefficiency scores (bad langs get more tokens)
    scores = df["inefficiency_score"].to_numpy()
    ideal = scores / scores.sum() * config.total_new_tokens
    alloc = np.floor(ideal).astype(int)

    shortfall = int(config.total_new_tokens - alloc.sum())
    if shortfall > 0:
        remainders = ideal - alloc
        idx = np.argsort(-remainders)[:shortfall]
        alloc[idx] += 1

    df["token_alloc"] = alloc
    df = df.sort_values("inefficiency_score", ascending=False).reset_index(drop=True)
    return df[
        [
            "language",
            "pieces_per_word",
            "chars_per_token",
            "unknown_rate",
            "ppw_n",
            "cpt_n",
            "unk_n",
            "I_lin",
            "inefficiency_score",
            "token_alloc",
        ]
    ]

