#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

dp=10
dm=32
nh=4
nl=1
hc=16
tp=11
ff=64

model_name=Signature

model_id="dp${dp}"

echo "Running with model_id: $model_id"

python -u run.py \
  --is_training 1 \
  --root_path ./asset_data/ \
  --data_path full_dataset_dow.csv \
  --model_id "$model_id" \
  --model "$model_name" \
  --data FULL \
  --data_pool "$dp" \
  --window_size 60 \
  --horizon 20 \
  --d_model "$dm" \
  --n_heads "$nh" \
  --num_layers "$nl" \
  --hidden_c "$hc" \
  --ff_dim "$ff" \
  --temperature "$tp" \
  --sig_input_dim 2 \
  --cross_sig_dim 1 \
  --des 'Exp' \
  --itr 3

wait


UDA_VISIBLE_DEVICES=0

dp=20
dm=32
nh=2
nl=1
hc=16
tp=13
ff=16

model_name=Signature

model_id="dp${dp}"

echo "Running with model_id: $model_id"

python -u run.py \
  --is_training 1 \
  --root_path ./asset_data/ \
  --data_path full_dataset_dow.csv \
  --model_id "$model_id" \
  --model "$model_name" \
  --data FULL \
  --data_pool "$dp" \
  --window_size 60 \
  --horizon 20 \
  --d_model "$dm" \
  --n_heads "$nh" \
  --num_layers "$nl" \
  --hidden_c "$hc" \
  --ff_dim "$ff" \
  --temperature "$tp" \
  --sig_input_dim 2 \
  --cross_sig_dim 1 \
  --des 'Exp' \
  --itr 3
