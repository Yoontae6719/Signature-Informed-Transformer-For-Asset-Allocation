# data_factory.py
import os, json
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import BDay

from data_provider.data_loader import Dataset_Sig, PrecomputedSigDataset

data_dict = {"FULL": Dataset_Sig}
_sig_scaler_global = None

def _build_loader(dataset, flag, args):
    if flag == "train":
        bs, sh, drop = args.batch_size, True, True
    elif flag == "val":
        bs, sh, drop = args.batch_size, False, True
    elif flag == "test":
        bs, sh, drop = 1, False, False
    else:
        raise ValueError(flag)
    loader = DataLoader(dataset, batch_size=bs, shuffle=sh,
                        num_workers=args.num_workers, drop_last=drop)
    print(f"[{flag.upper()}] len={len(dataset)} | batch_size={bs}")
    return dataset, loader

def data_provider(args, flag, scale):

    #  1. pre‑computed (see 0_get_sig_data_all.py)
    if getattr(args, "precomp_root", None):
        cache_dir = os.path.join(args.precomp_root, f"pool_{args.data_pool}")
        dataset = PrecomputedSigDataset(cache_dir, flag)
        return _build_loader(dataset, flag, args)

    global _sig_scaler_global
    csv_path = os.path.join(args.root_path, args.data_path)
    combined_df = (
        pd.read_csv(csv_path, parse_dates=["Date"])
          .set_index("Date")
    ).iloc[:, :args.data_pool]

    train_df = combined_df.loc["2000-01-01":"2016-12-31"]
    val_df   = combined_df.loc["2017-01-01":"2019-12-31"]
    TEST_START, TEST_END = "2020-01-01", "2024-12-31"
    #ctx_start = (pd.to_datetime(TEST_START) - BDay(args.window_size + 1)).strftime("%Y-%m-%d")
    ix        = combined_df.index.get_loc(TEST_START)
    ctx_start = combined_df.index[ix - (args.window_size + 1)]
    test_df  = combined_df.loc[ctx_start:TEST_END]

    df_map = {"train": train_df, "val": val_df, "test": test_df}
    df_use = df_map[flag]

    if flag == "test":
        dataset = Dataset_Sig(df_use, df_use,
                              args.window_size, args.horizon,
                              sig_scaler=None,
                              pred_start=TEST_START, pred_end=TEST_END)
    else:
        dataset = Dataset_Sig(df_use, df_use,
                              args.window_size, args.horizon,
                              sig_scaler=None)

    if flag == "train":
        if _sig_scaler_global is None:
            feats = [np.concatenate([dataset[i]["x_sigs"].ravel(),
                                     dataset[i]["cross_sigs"].ravel()])
                     for i in range(len(dataset))]
            _sig_scaler_global = StandardScaler().fit(np.vstack(feats))
        dataset.set_sig_scaler(_sig_scaler_global)
    else:
        if _sig_scaler_global is None:
            raise RuntimeError("call train first to fit scaler")
        dataset.set_sig_scaler(_sig_scaler_global)

    return _build_loader(dataset, flag, args)
