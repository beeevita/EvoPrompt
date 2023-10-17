#!/bin/bash

set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8  
export CUDA_VISIBLE_DEVICES=0

BUDGET=10
POPSIZE=10
SEED=5
TEMPLATE=v1
INITIAL_MODE=para_topk
LLM_TYPE=turbo

for DATASET in cr
do
OUT_PATH=outputs/cls/$DATASET/alpaca/all/de/bd${BUDGET}_top${POPSIZE}_${INITIAL_MODE}_init/${TEMPLATE}/$LLM_TYPE
for SEED in 5 10 15
do
python run.py \
    --seed $SEED \
    --dataset $DATASET \
    --task cls \
    --batch-size 32 \
    --prompt-num 0 \
    --sample_num 500 \
    --language_model alpaca \
    --budget $BUDGET \
    --popsize $POPSIZE \
    --position demon \
    --evo_mode de \
    --llm_type $LLM_TYPE \
    --setting default \
    --write_step 5 \
    --initial all \
    --initial_mode $INITIAL_MODE \
    --template $TEMPLATE \
    --output $OUT_PATH/seed$SEED \
    --cache_path data/cls/$DATASET/seed${SEED}/prompts_batched.json \
    --dev_file ./data/cls/$DATASET/seed${SEED}/dev.txt
done
python get_result.py -p $OUT_PATH > $OUT_PATH/result.txt
done