#!/bin/bash

set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8  
export CUDA_VISIBLE_DEVICES=0

BUDGET=10
POPSIZE=10
GA=topk
LLM_TYPE=davinci

for dataset in asset
do
OUT_PATH=outputs/sim/$dataset/alpaca/all/ga/bd${BUDGET}_top${POPSIZE}_para_topk_init/$GA/$LLM_TYPE
for SEED in 5 10 15
do
python run.py \
    --seed $SEED \
    --dataset $dataset \
    --task sim \
    --batch-size 32 \
    --prompt-num 0 \
    --sample_num 100 \
    --language_model alpaca \
    --budget $BUDGET \
    --popsize $POPSIZE \
    --position pre \
    --evo_mode ga \
    --llm_type $LLM_TYPE \
    --initial all \
    --initial_mode para_topk \
    --ga_mode $GA \
    --cache_path data/sim/$dataset/seed${SEED}/prompts_batched.json \
    --output $OUT_PATH/seed$SEED
done
python get_result.py -p $OUT_PATH > $OUT_PATH/result.txt
done