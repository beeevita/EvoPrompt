#!/bin/bash

set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8  
export CUDA_VISIBLE_DEVICES=0

BUDGET=10

POPSIZE=10
TEMPLATE=v1
initial=all
LLM_TYPE=davinci

for dataset in asset
do
OUT_PATH=outputs/sim/$dataset/alpaca/all/de/bd${BUDGET}_top${POPSIZE}_para_topk_init/${TEMPLATE}/$LLM_TYPE
mkdir -p $OUT_PATH
for SEED in 5 10 15
do
python run.py \
    --seed $SEED \
    --dataset $dataset \
    --task sim \
    --batch-size 20 \
    --prompt-num 0 \
    --sample_num 100 \
    --language_model alpaca \
    --budget $BUDGET \
    --popsize $POPSIZE \
    --position pre \
    --evo_mode de \
    --llm_type $LLM_TYPE \
    --initial $initial \
    --setting default \
    --initial_mode para_topk \
    --cache_path data/sim/$dataset/seed${SEED}/prompts_batched.json \
    --template $TEMPLATE \
    --output $OUT_PATH/seed$SEED
done
python get_result.py -p $OUT_PATH > $OUT_PATH/result.txt
done