# data_loader.py

import os, json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features   # ★ NEW



class Dataset_Sig(Dataset):

    def __init__(self,
                 scaled_df: pd.DataFrame,
                 unscaled_df: pd.DataFrame,
                 window_size: int = 60,
                 horizon: int = 20,
                 sig_scaler: StandardScaler | None = None,
                 pred_start: str | None = None,
                 pred_end: str | None = None):
        super().__init__()
        self.prices        = scaled_df.values.astype(np.float32)
        self.raw_prices    = unscaled_df.values.astype(np.float32)
        self.dates         = scaled_df.index
        self.W             = window_size
        self.H             = horizon
        self.D             = self.prices.shape[1]
        self.sig_scaler    = sig_scaler
        self.num_assets    = self.D
        self.pred_start = pd.to_datetime(pred_start) if pred_start else None
        self.pred_end   = pd.to_datetime(pred_end)   if pred_end   else None
        self._build_valid_indices()

    @staticmethod
    def _sig_level2(p):
        return np.array([p[-1] - p[0],
                         np.sum(p[:-1] * np.diff(p))],
                        dtype=np.float32)

    @staticmethod
    def _sig_cross(pj, pk):
        dj = np.diff(pj, prepend=pj[0])
        dk = np.diff(pk, prepend=pk[0])
        return np.array([np.sum(np.cumsum(dj)[:-1] * dk[1:])], dtype=np.float32)

    def _build_valid_indices(self):
        max_start = len(self.prices) - self.W - self.H
        idxs = []
        for i in range(max_start):
            first_pred = self.dates[i + self.W] #  20250723 self.dates[i + self.W + 1]
            last_pred  = self.dates[i + self.W + self.H]
            if ((self.pred_start is None or first_pred >= self.pred_start) and
                (self.pred_end   is None or last_pred  <= self.pred_end)):
                idxs.append(i)
        self.valid_idx = np.asarray(idxs, dtype=np.int32)
        if len(self.valid_idx) == 0:
            raise RuntimeError("No sample satisfies pred_start / pred_end")

    def set_sig_scaler(self, scaler: StandardScaler):
        self.sig_scaler = scaler

    def _scale_sigs(self, x_sigs, cross_sigs):
        if self.sig_scaler is None:
            return x_sigs, cross_sigs
        H, D, _ = x_sigs.shape
        flat = np.concatenate([x_sigs.ravel(), cross_sigs.ravel()])[None, :]
        flat_std = self.sig_scaler.transform(flat).ravel()
        n_x = H * D * 2
        return (flat_std[:n_x].reshape(H, D, 2).astype(np.float32),
                flat_std[n_x:].reshape(H, D, D, 1).astype(np.float32))

    #  sampler
    def _build_sample(self, start_idx: int):
        w0, w1 = start_idx, start_idx + self.W
        h0, h1 = w1, w1 + self.H

        # Step 1. signature 
        x_sigs, cross_sigs = [], []
        for step in range(self.H):
            p_slice = self.prices[w0 + step : w1 + step]   # (W+1, D)
            x_sigs.append(
                np.stack([self._sig_level2(p_slice[:, j]) for j in range(self.D)], axis=0)
            )
            cross_step = []
            for j in range(self.D):
                row = []
                for k in range(self.D):
                    row.append(np.array([0.0], dtype=np.float32) if j == k
                               else self._sig_cross(p_slice[:, j], p_slice[:, k]))
                cross_step.append(np.stack(row, axis=0))
            cross_sigs.append(np.stack(cross_step, axis=0))

        x_sigs     = np.stack(x_sigs, axis=0)                 # (H, D, 2)
        cross_sigs = np.stack(cross_sigs, axis=0)             # (H, D, D, 1)
        x_sigs, cross_sigs = self._scale_sigs(x_sigs, cross_sigs)

        # Step 2 future returns & date strings 
        fut_ret, dates = [], []
        for t in range(h0, h1):
            p_now, p_next = self.raw_prices[t], self.raw_prices[t + 1]
            fut_ret.append((p_next - p_now) / (p_now + 1e-8))
            dates.append(self.dates[t].strftime("%Y-%m-%d")) #  20250723 self.dates[i + self.W + 1]
        fut_ret = np.stack(fut_ret, axis=0).astype(np.float32)  # (H, D)

        # Step 3. time embeddings
        date_feats = time_features(pd.to_datetime(dates), freq="B").T.astype(np.float32)  # (H, F)

        return {
            "x_sigs": x_sigs,               # (H, D, 2)
            "cross_sigs": cross_sigs,       # (H, D, D, 1)
            "future_return_unscaled": fut_ret,   # (H, D)
            "dates_horizon": dates,         # list[str]
            "date_feats": date_feats        # (H, F)
        }

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        return self._build_sample(int(self.valid_idx[idx]))


# =================================================================
# PrecomputedSigDataset  (date_feats 즉석 생성)  ───────────────────
# =================================================================
class PrecomputedSigDataset(Dataset):
    def __init__(self, cache_dir: str, split: str):
        print("USE PrecomputedSigDataset")

        
        if split not in ("train", "val", "test", "full"):
            raise ValueError("split must be train/val/test/full")

        self.x     = np.memmap(os.path.join(cache_dir, f"{split}_x.npy"),
                               dtype="float32", mode="r")
        self.cross = np.memmap(os.path.join(cache_dir, f"{split}_cross.npy"),
                               dtype="float32", mode="r")
        self.ret   = np.memmap(os.path.join(cache_dir, f"{split}_ret.npy"),
                               dtype="float32", mode="r")
        self.dates = np.load(os.path.join(cache_dir, f"{split}_dates.npy"),
                             allow_pickle=True)

        with open(os.path.join(cache_dir, f"{split}_meta.json")) as fp:
            meta = json.load(fp)

        self.N, self.H, self.D = meta["N"], meta["H"], meta["D"]
        self.num_assets = self.D
        self.x     = self.x.reshape(self.N, self.H, self.D, 2)
        self.cross = self.cross.reshape(self.N, self.H, self.D, self.D, 1)
        self.ret   = self.ret.reshape(self.N, self.H, self.D)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        dates = self.dates[idx].tolist()
        date_feats = time_features(pd.to_datetime(dates), freq="B").T.astype(np.float32)
        return {
            "x_sigs"     : self.x[idx],
            "cross_sigs" : self.cross[idx],
            "future_return_unscaled": self.ret[idx],
            "dates_horizon": dates,
            "date_feats" : date_feats
        }
