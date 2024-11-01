#!/bin/bash
for ((seed=0; seed<$2; seed++)); do
    python method.py --run experiments/$1 --train_only photo --seed $seed --method_loss 1 --epochs 300 --backbone $3 --dataset PACS --gpu $4 &
done
wait
