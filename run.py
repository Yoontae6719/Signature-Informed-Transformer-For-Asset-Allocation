import argparse
import torch
from exp.exp_main import EXP_main
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Signature-informed Transformers for Multi-Asset Trading')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='SIT')


    # data loader
    parser.add_argument('--data', type=str, required=True, default='FULL', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./asset_data/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='full_dataset.csv', help='data csv file')
    parser.add_argument('--freq', type=str, default='b',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--data_pool', type=int, default=10, help='Trading dimensions')
    parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--flag', type=int, default=1, help='data csv file')
    parser.add_argument('--time_feat_dim', type=int, default=3, help='time_feat_dim')

    parser.add_argument('--precomp_root', type=str, default='./signature_cache_6020', help='root dir of pre-computed signature cache (empty string to disable)')

    # forecasting task
    parser.add_argument('--window_size', type=int, default=40, help='input sequence length')
    parser.add_argument('--horizon', type=int, default=1, help='prediction sequence length')

    # model define
    ## Default params
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--sig_input_dim', type=int, default=2, help='Degree of Signature')
    parser.add_argument('--cross_sig_dim', type=int, default=1, help='Degree of Cross-Signature')
    parser.add_argument('--hidden_c', type=int, default=16, help='Cross signature score')

    parser.add_argument('--max_position', type=int, default=5, help='Buy and Sell position')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    ## Transformer params
    parser.add_argument('--ff_dim', type=int, default=64, help='dimension of FF')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--num_layers', type=int, default=2, help='num of layers')

    ## Loss params
    parser.add_argument('--lambda_cond', type=float, default=1, help='conditional lambda')
    parser.add_argument('--temperature', type=float, default=1.3, help='Portfolio temper')
    parser.add_argument('--trade_cost_bps', type=float, default=0.0, help='0.05 % cost in bps')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = EXP_main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_dp{}_sl{}_pl{}_dm{}_nh{}_nl{}_si{}_sc{}_hc{}_mp{}_ff{}_de{}_tp{}_cb{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.data_pool,
                args.window_size,
                args.horizon,
                args.d_model,
                args.n_heads,
                args.num_layers,
                args.sig_input_dim,
                args.cross_sig_dim,
                args.hidden_c,
                args.max_position,
                args.ff_dim, 
                args.des, args.temperature, args.trade_cost_bps, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.eval(setting, True)


            torch.cuda.empty_cache()
    else:
        pass
