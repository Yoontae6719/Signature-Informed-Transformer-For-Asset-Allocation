import os, time, warnings, math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate

warnings.filterwarnings("ignore")
REBAL_DATES_STR = [
    "2020-01-03", "2020-01-31", "2020-03-02", "2020-03-30", "2020-04-28",
    "2020-05-27", "2020-06-24", "2020-07-23", "2020-08-20", "2020-09-18",
    "2020-10-16", "2020-11-13", "2020-12-14", "2021-01-13", "2021-02-11",
    "2021-03-12", "2021-04-12", "2021-05-10", "2021-06-08", "2021-07-07",
    "2021-08-04", "2021-09-01", "2021-09-30", "2021-10-28", "2021-11-26",
    "2021-12-27", "2022-01-25", "2022-02-23", "2022-03-23", "2022-04-21",
    "2022-05-19", "2022-06-17", "2022-07-19", "2022-08-16", "2022-09-14",
    "2022-10-12", "2022-11-09", "2022-12-08", "2023-01-09", "2023-02-07",
    "2023-03-08", "2023-04-05", "2023-05-04", "2023-06-02", "2023-07-03",
    "2023-08-01", "2023-08-29", "2023-09-27", "2023-10-25", "2023-11-22",
    "2023-12-21", "2024-01-23", "2024-02-21", "2024-03-20", "2024-04-18",
    "2024-05-16", "2024-06-14", "2024-07-16", "2024-08-13", "2024-09-11",
    "2024-10-09", "2024-11-06", "2024-12-05", 
]
REBAL_DATES       = pd.to_datetime(REBAL_DATES_STR)
REBAL_DATE_SET    = set(REBAL_DATES_STR)   

def decision_focused_cvar_loss(
    model:      nn.Module,
    x_sigs:     torch.Tensor,
    cross_sigs: torch.Tensor,
    date_feats: torch.Tensor,
    fut_ret:    torch.Tensor,
    alpha:      float  = 0.95,
    temperature: float = 1.0,) -> Tuple[torch.Tensor, float]:
    
    _, mu_hat = model(x_sigs, cross_sigs, date_feats)              # (B,H,D)
    B, H, D   = mu_hat.shape
    weights   = F.softmax(mu_hat / temperature, dim=-1)            # (B,H,D)

    port_ret  = torch.sum(weights * fut_ret, dim=-1)               # (B,H)
    port_loss = -port_ret

    VaR   = torch.quantile(port_loss, alpha, dim=1, keepdim=True)  # (B,1)
    excess= F.relu(port_loss - VaR)
    cvar  = VaR.squeeze(-1) + excess.mean(dim=1) / (1.0 - alpha)   # (B,)

    loss      = cvar.mean()
    avg_cvar  = cvar.mean().item()
    return loss, avg_cvar


class EXP_main(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        for k, v in dict(cvar_alpha=0.95, softmax_tau=1.0,
                         train_epochs=30, time_feat_dim=3).items():
            if not hasattr(args, k):
                setattr(args, k, v)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, scale):
        return data_provider(self.args, flag, scale)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)


    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train", scale=True)
        vali_data,  vali_loader  = self._get_data(flag="val",   scale=True)
        test_data,  test_loader  = self._get_data(flag="test",  scale=False)

        ckpt_dir = Path(self.args.checkpoints) / setting
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim    = self._select_optimizer()

        time_now   = time.time()
        train_steps = len(train_loader)

        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_start = time.time()

            losses, cvars = [], []
            for i, batch in enumerate(train_loader):
                model_optim.zero_grad()

                x_sigs     = batch["x_sigs"].to(self.device)
                cross_sigs = batch["cross_sigs"].to(self.device)
                date_feats = batch["date_feats"].to(self.device)   
                fut_ret    = batch["future_return_unscaled"].to(self.device)

                loss, batch_cvar = decision_focused_cvar_loss(
                    model        = self.model,
                    x_sigs       = x_sigs,
                    cross_sigs   = cross_sigs,
                    date_feats   = date_feats,      
                    fut_ret      = fut_ret,
                    alpha        = self.args.cvar_alpha,
                    temperature  = self.args.temperature
                )
                loss.backward()
                model_optim.step()

                losses.append(loss.item());  cvars.append(batch_cvar)

                
                if (i + 1) % 100 == 0 or (i + 1) == train_steps:
                    speed = (time.time() - time_now) / (i + 1)
                    left  = speed * ((self.args.train_epochs - epoch) *
                                     train_steps - (i + 1))
                    print(f"\titers {i+1}/{train_steps}, epoch {epoch+1} | "
                          f"loss {loss.item():.6f} | CVaR {batch_cvar:.6f} "
                          f"| {speed:.3f}s/iter, ETA {left:.1f}s")
            print(f"[Epoch {epoch+1}] mean_loss={np.mean(losses):.6f} "
                  f"mean_CVaR={np.mean(cvars):.6f} "
                  f"({time.time()-epoch_start:.1f}s)")

           
            vali_loss = self._vali(vali_loader)
            test_loss = self._vali(test_loader, is_vali=False)
            print(f"    >> Vali_CVaR={vali_loss:.6f} | Test_CVaR={test_loss:.6f}")

            early_stopping(vali_loss, self.model, str(ckpt_dir))
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        
        self.model.load_state_dict(torch.load(ckpt_dir / "checkpoint.pth"))
        return self.model

    def _vali(self, loader, is_vali=True):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in loader:
                x_sigs     = batch["x_sigs"].to(self.device)
                cross_sigs = batch["cross_sigs"].to(self.device)
                date_feats = batch["date_feats"].to(self.device)   
                fut_ret    = batch["future_return_unscaled"].to(self.device)

                loss, _ = decision_focused_cvar_loss(
                    model        = self.model,
                    x_sigs       = x_sigs,
                    cross_sigs   = cross_sigs,
                    date_feats   = date_feats,      
                    fut_ret      = fut_ret,
                    alpha        = self.args.cvar_alpha,
                    temperature  = self.args.temperature
                )
                losses.append(loss.item())
        self.model.train()
        return float(np.mean(losses))


    def eval(self, setting, load=True):
        test_data, test_loader = self._get_data(flag="test", scale=False)
        asset_names = [f"Asset_{i}" for i in range(test_loader.dataset.num_assets)]

        if load:
            ckpt = Path(self.args.checkpoints) / setting / "checkpoint.pth"
            self.model.load_state_dict(torch.load(ckpt))
            print(f"[Eval] Loaded best checkpoint → {ckpt}")

        self.model.eval()
        out_dir = Path("./results"); out_dir.mkdir(exist_ok=True)
        csv_path = out_dir / f"{setting}_test_positions.csv"
        fig_path = out_dir / f"{setting}_test_equity_curve.png"
        met_path = out_dir / f"{setting}_test_metrics.csv"

        record = {}
        with torch.no_grad():
            for batch in test_loader:
                x_sigs     = batch["x_sigs"].to(self.device)
                cross_sigs = batch["cross_sigs"].to(self.device)
                date_feats = batch["date_feats"].to(self.device)
                fut_ret    = batch["future_return_unscaled"].to(self.device)  # (B,H,D)
                raw_dates  = batch["dates_horizon"]

                dates = [list(col) for col in zip(*raw_dates)] if isinstance(
                            raw_dates[0], (list, tuple, np.ndarray)) else [raw_dates]

                _, mu_hat = self.model(x_sigs, cross_sigs, date_feats)        # (B,H,D)
                pred_w    = F.softmax(mu_hat / self.args.temperature, dim=-1) # (B,H,D)

                B, H, D = fut_ret.shape
                for b in range(B):
                    for t in range(H):
                        d_str = dates[b][t]
                        if d_str in record:
                            continue
                        record[d_str] = (
                            pred_w[b, t].cpu().numpy(),    # weight_vec
                            fut_ret[b, t].cpu().numpy()    # fut_ret_vec
                        )
                        # =====================================================

        sorted_dates = sorted(record.keys(), key=pd.to_datetime)
        initial_cap  = 10_000.0
        capital      = initial_cap
        equity_curve = [capital]
        daily_rets   = []
        rows         = []
        cost_rate = self.args.trade_cost_bps * 1e-4          # # add featrue for cost
        prev_w    = np.zeros(len(asset_names), dtype=float)  # # add featrue for cost
    
        current_w = None   

        for d_str in sorted_dates:
            pred_w, fut_ret_vec = record[d_str]
            cost = 0.0   # add featrue for cost


            if (current_w is None) or (d_str in REBAL_DATE_SET):
                current_w = pred_w
                
                turnover = np.abs(current_w - prev_w).sum()    # add featrue for cost
                cost     = turnover * cost_rate * capital      # add featrue for cost
                current_w = pred_w           # add featrue for cost
                prev_w    = current_w.copy() # add featrue for cost       

            prev_cap = capital
            pnl      = float(np.dot(current_w, fut_ret_vec)) * prev_cap
            pnl     -= cost                                     # add featrue for cost
            capital += pnl

            equity_curve.append(capital)
            daily_rets.append(pnl / (prev_cap + 1e-12))

            #row = {"Date": d_str,
            #       "DailyPnL": pnl,
            #       "CumulativeCapital": capital}
            
            row = {"Date": d_str,
                   "DailyPnL": pnl,
                   "Cost": cost,
                   "Turnover": turnover if cost > 0 else 0, # 리밸런싱 날에만 기록
                   "CumulativeCapital": capital}
            
            for i, name in enumerate(asset_names):
                row[f"Weight_{name}"] = float(current_w[i])
            
            rows.append(row)


        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[Eval] Position log → {csv_path}")

        eq   = np.asarray(equity_curve)
        rets = np.asarray(daily_rets)
        ann  = 252
        sharpe   = (rets.mean() / (rets.std() + 1e-12)) * math.sqrt(ann)
        neg_std  = rets[rets < 0].std() + 1e-12
        sortino  = (rets.mean() / neg_std) * math.sqrt(ann)
        peak     = np.maximum.accumulate(eq)
        max_dd   = np.max((peak - eq) / (peak + 1e-12))
        fin_wf   = eq[-1] / eq[0]
        ann_ret  = fin_wf**(ann / len(rets)) - 1.0
        ann_vol  = rets.std() * math.sqrt(ann)
        win_rate = (rets > 0).mean()

        pd.DataFrame([dict(Sharpe=sharpe, Sortino=sortino,
                           MaxDrawdown=max_dd, AnnualReturn=ann_ret,
                           AnnualVol=ann_vol, FinalWealthFactor=fin_wf,
                           WinRate=win_rate)]
                     ).to_csv(met_path, index=False)
        print(f"[Eval] Metrics → {met_path}")

        plt.figure(figsize=(10, 5))
        plt.plot(eq, label="Equity")
        plt.title("Equity Curve – Test (Rebalanced on specified dates)")
        plt.xlabel("Step"); plt.ylabel("Capital")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(fig_path); plt.close()
        print(f"[Eval] Equity curve → {fig_path}")
