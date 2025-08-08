import os, json, joblib
import numpy as np
import pandas as pd
from tqdm import trange
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import BDay

from data_provider.data_loader import Dataset_Sig

CONFIGS = [(60, 20)]

CSV_PATH     = "./asset_data/full_dataset.csv"
DATA_POOLS   = [40, 50, 30]
SIG_SAMPLE_N = None
TEST_START   = "2020-01-01"
TEST_END     = "2024-12-31"

def fit_sig_scaler(ds: Dataset_Sig, max_samples=None):
    rows = []
    N = len(ds) if max_samples is None else min(len(ds), max_samples)
    for i in trange(N, desc="fit scaler"):
        it = ds[i]
        rows.append(np.concatenate([it["x_sigs"].ravel(),
                                    it["cross_sigs"].ravel()]))
    return StandardScaler().fit(np.vstack(rows))

def save_split(ds: Dataset_Sig, name: str, out_dir: str):
    N, H, D = len(ds), ds.H, ds.D
    print(f"[{name.upper()}] N={N}, H={H}, D={D}")
    os.makedirs(out_dir, exist_ok=True)
    def _mm(path, shape): return np.memmap(path, dtype="float32", mode="w+", shape=shape)

    mm_x     = _mm(os.path.join(out_dir, f"{name}_x.npy"),      (N, H, D, 2))
    mm_cross = _mm(os.path.join(out_dir, f"{name}_cross.npy"), (N, H, D, D, 1))
    mm_ret   = _mm(os.path.join(out_dir, f"{name}_ret.npy"),   (N, H, D))
    date_arr = []

    for i in trange(N, desc=f"save {name}"):
        it = ds[i]
        mm_x[i]     = it["x_sigs"]
        mm_cross[i] = it["cross_sigs"]
        mm_ret[i]   = it["future_return_unscaled"]
        date_arr.append(it["dates_horizon"])

    mm_x.flush(); mm_cross.flush(); mm_ret.flush()
    np.save(os.path.join(out_dir, f"{name}_dates.npy"), np.array(date_arr, dtype=object))
    json.dump({"N": N, "H": H, "D": D},
              open(os.path.join(out_dir, f"{name}_meta.json"), "w"))
    print(f"[{name.upper()}] saved → {out_dir}")

df_full_original = pd.read_csv(CSV_PATH, parse_dates=["Date"]).set_index("Date")


for W, H in CONFIGS:
    WINDOW_SIZE = W
    HORIZON = H
    
    SAVE_ROOT = f"./signature_cache_{W}{H}"
    CTX_PAD   = WINDOW_SIZE + 1
    
    print(f"\n\n{'='*25} PROCESSING CONFIG: WINDOW={WINDOW_SIZE}, HORIZON={HORIZON} {'='*25}")

    for pool in DATA_POOLS:
        print(f"\n========== DATA_POOL = {pool} ==========")
        df = df_full_original.iloc[:, :pool]

        train_df = df.loc["2000-01-01":"2016-12-31"]
        val_df   = df.loc["2017-01-01":"2019-12-31"]
        ctx_start = (pd.to_datetime(TEST_START) - BDay(CTX_PAD)).strftime("%Y-%m-%d")
        test_df  = df.loc[ctx_start:TEST_END]
        full_df  = df

        train_ds = Dataset_Sig(train_df, train_df, WINDOW_SIZE, HORIZON)
        val_ds   = Dataset_Sig(val_df, val_df, WINDOW_SIZE, HORIZON)
        test_ds  = Dataset_Sig(test_df, test_df, WINDOW_SIZE, HORIZON,
                               pred_start=TEST_START, pred_end=TEST_END)
        full_ds  = Dataset_Sig(full_df, full_df, WINDOW_SIZE, HORIZON)

        scaler = fit_sig_scaler(train_ds, SIG_SAMPLE_N)
        for ds in (train_ds, val_ds, test_ds, full_ds):
            ds.set_sig_scaler(scaler)

        out_dir = os.path.join(SAVE_ROOT, f"pool_{pool}")
        save_split(train_ds, "train", out_dir)
        save_split(val_ds,   "val",   out_dir)
        save_split(test_ds,  "test",  out_dir)
        save_split(full_ds,  "full",  out_dir)
        joblib.dump(scaler, os.path.join(out_dir, "signature_scaler.pkl"))
        print(f"[Scaler] saved → {out_dir}/signature_scaler.pkl")

print("\nComplete.")
