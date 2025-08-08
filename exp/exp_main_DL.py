from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
import pandas as pd

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
    "2024-10-09", "2024-11-06", "2024-12-05"
]
REBAL_DATES       = pd.to_datetime(REBAL_DATES_STR)
REBAL_DATE_SET    = set(REBAL_DATES_STR)   


warnings.filterwarnings('ignore')

import pandas as pd, math, matplotlib.pyplot as plt
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    print("[Warn] cvxpy not found → fallback to naïve closed‑form MVO")
    HAS_CVXPY = False


class EXP_main(Exp_Basic):
    def __init__(self, args):
        super(EXP_main, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()


        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data('test')
        if test:
            print('loading trained checkpoint...')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints', setting, 'checkpoint.pth')))

        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x      = batch_x.float().to(self.device)
                batch_y      = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)
                
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim   = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                orig_shape = outputs.shape
                outputs_2d = outputs.reshape(-1, orig_shape[-1])
                batch_y_2d = batch_y.reshape(-1, orig_shape[-1])
                
                outputs_inv = test_data.inverse_transform(outputs_2d).reshape(orig_shape)
                batch_y_inv = test_data.inverse_transform(batch_y_2d).reshape(orig_shape)
                
                outputs = outputs_inv[:, :, f_dim:]
                batch_y = batch_y_inv[:, :, f_dim:]
                
                preds.append(outputs)
                trues.append(batch_y)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        folder_path = os.path.join('./results_bench', setting)
        os.makedirs(folder_path, exist_ok=True)
        
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)


        horizon  = self.args.pred_len
        n_assets = preds.shape[-1]
        
        num_test_days = preds.shape[0] + horizon - 1
        all_test_dates = test_data.dates[test_data.seq_len : test_data.seq_len + num_test_days]

        all_test_dates_str = np.datetime_as_string(all_test_dates, unit='D') 
        rebal_indices = [i for i, date_str in enumerate(all_test_dates_str) if date_str in REBAL_DATE_SET]


        weights_cvar, daily_rets_cvar, window_rets_cvar = [], [], []
        daily_dates = []
        rebalancing_dates = []

        # hyper-parameters
        cvar_alpha  = getattr(self.args, "cvar_alpha", 0.95)
        lmbd_cvar   = getattr(self.args, "cvar_lambda", 1.0)
        
        for i in range(len(rebal_indices)):
            day_idx = rebal_indices[i]

            if i + 1 < len(rebal_indices):
                next_rebal_idx = rebal_indices[i+1]
            else:
                next_rebal_idx = num_test_days

            pred_sample_idx = min(day_idx, preds.shape[0] - 1)
            pred_scn = preds[pred_sample_idx]
            
            days_in_this_period = next_rebal_idx - day_idx
            if days_in_this_period <= 0: continue 

            if day_idx >= trues.shape[0]:
                true_scn_list = []
                for j in range(days_in_this_period):
                    d = day_idx + j
                    sample_idx = trues.shape[0] - 1
                    offset = d - sample_idx
                    if offset >= trues.shape[1]: continue 
                    true_scn_list.append(trues[sample_idx, offset, :])
                true_scn = np.array(true_scn_list)
            else:
                true_scn = trues[day_idx][:days_in_this_period]
            
            if true_scn.shape[0] == 0: continue

            current_dates = all_test_dates[day_idx : next_rebal_idx]

            rebalancing_dates.append(current_dates[0])
            
            mu = pred_scn.mean(axis=0)
            
            from sklearn.covariance import LedoitWolf
            Sigma = LedoitWolf().fit(pred_scn).covariance_
            Sigma = (Sigma + Sigma.T)/2 + 1e-6*np.eye(n_assets)
            
            if HAS_CVXPY:
                T_    = pred_scn.shape[0]
                w     = cp.Variable(n_assets)
                z     = cp.Variable(T_)
                t     = cp.Variable()
                port  = pred_scn @ w
                obj   = cp.Minimize(-mu @ w + lmbd_cvar * (t + (1/((1-cvar_alpha)*T_)) * cp.sum(z)))
                cons  = [cp.sum(w) == 1, w >= 0,
                         z >= 0, z >= -port - t]
                cp.Problem(obj, cons).solve(solver=cp.SCS, verbose=False)
                w_cvar = np.asarray(w.value).flatten()
                if w_cvar is None or np.any(np.isnan(w_cvar)):
                    w_cvar = np.full(n_assets, 1.0 / n_assets)
            else:
                w_cvar = np.full(n_assets, 1.0 / n_assets)
            
            weights_cvar.append(w_cvar)
            
            port_path_cvar = true_scn @ w_cvar
            daily_rets_cvar.extend(port_path_cvar.tolist())
            daily_dates.extend(current_dates.tolist())
            window_rets_cvar.append(np.prod(1 + port_path_cvar) - 1.0)
            
        if not daily_dates:
            print("[Error] No daily returns were generated. Backtest failed.")
            return

        rets_cvar_ser = pd.Series(daily_rets_cvar, index=pd.to_datetime(daily_dates))
        rets_cvar_ser = rets_cvar_ser[~rets_cvar_ser.index.duplicated(keep="last")].sort_index()

        
        daily_dates     = rets_cvar_ser.index
        daily_rets_cvar = rets_cvar_ser.values
        
        weights_cvar    = np.asarray(weights_cvar)
        daily_rets_cvar = np.asarray(daily_rets_cvar)
        window_rets_cvar= np.asarray(window_rets_cvar)
        daily_dates     = pd.to_datetime(daily_dates)
        
        # 결과 저장
        rebal_df = pd.DataFrame(rebalancing_dates, columns=["RebalancingDate"])
        rebal_df.to_csv(os.path.join(folder_path, "rebalancing_dates.csv"), index=False)
        print(f"[Test] Rebalancing dates saved → {folder_path}/rebalancing_dates.csv")
        
        np.save(os.path.join(folder_path, "weights_cvar.npy"),   weights_cvar)
        np.save(os.path.join(folder_path, "daily_ret_cvar.npy"), daily_rets_cvar)
        np.save(os.path.join(folder_path, "window_ret_cvar.npy"),window_rets_cvar)
        print(f"[Test] posterior optimisation save → {folder_path}")
        
        pd.DataFrame(
            weights_cvar,
            columns=[f"Asset_{i+1}" for i in range(n_assets)]
        ).to_csv(os.path.join(folder_path, "weights_cvar.csv"), index=False)
        

        initial_cap = 10_000.0
        equity_cvar = initial_cap * np.cumprod(1 + daily_rets_cvar)
        
        plt.figure(figsize=(9, 4.5))
        plt.plot(daily_dates, equity_cvar, label="Mean-CVaR")
        plt.title(f"Equity Curve – Daily (rebalance on specified dates)")
        plt.xlabel("Trading Day")
        plt.ylabel("Capital")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(folder_path, "equity_curve.png"))
        plt.close()
        print(f"[Test] Equity curve saved → {folder_path}/equity_curve.png")
        
        equity_df = pd.DataFrame({
            "Date"        : daily_dates,
            "Equity_CVaR" : equity_cvar
        })
        equity_df.to_csv(os.path.join(folder_path, "equity_curve.csv"), index=False)
        print(f"[Test] Equity curve (CSV) saved → {folder_path}/equity_curve.csv")

        def compute_metrics(rets):
            if len(rets) == 0:
                return dict.fromkeys(['Sharpe', 'Sortino', 'MaxDrawdown', 'AnnualReturn', 'AnnualVol', 'FinalWealthFactor', 'WinRate'], 0)
            ann_factor = 252
            eq_curve   = np.cumprod(1 + rets)
            peak       = np.maximum.accumulate(eq_curve)
            max_dd     = np.max((peak - eq_curve) / (peak + 1e-12))
            neg_std    = rets[rets < 0].std() + 1e-12
            return dict(
                Sharpe          = (rets.mean() / (rets.std()+1e-12)) * math.sqrt(ann_factor),
                Sortino         = (rets.mean() / neg_std)           * math.sqrt(ann_factor),
                MaxDrawdown     = max_dd,
                AnnualReturn    = (eq_curve[-1] ** (ann_factor/len(rets))) - 1.0,
                AnnualVol       = rets.std() * math.sqrt(ann_factor),
                FinalWealthFactor = eq_curve[-1],
                WinRate         = (rets > 0).mean()
            )
        
        met_cvar = compute_metrics(daily_rets_cvar)
        pd.DataFrame([met_cvar]).to_csv(os.path.join(folder_path, "metrics_cvar.csv"), index=False)
        print(f"[Test] Metrics saved → {folder_path}/metrics_cvar.csv")
        return
