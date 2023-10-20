#!/bin/bash

set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8  

BUDGET=10

POPSIZE=10
TEMPLATE=v1
initial=all

for dataset in sam
do
OUT_PATH=outputs/sum/$dataset/alpaca/all/de/bd${BUDGET}_top${NUM_OPOPSIZEF_MANUAL}_para_topk_init/${TEMPLATE}/davinci
for SEED in 5 10 15
do
python run.py \
    --seed $SEED \
    --do_test \
    --dataset $dataset \
    --task sum \
    --batch-size 20 \
    --prompt-num 0 \
    --sample_num 100 \
    --language_model alpaca \
    --budget $BUDGET \
    --popsize $POPSIZE \
    --position pre \
    --evo_mode de \
    --llm_type davinci \
    --initial all \
    --initial_mode para_topk \
    --cache_path data/sum/$dataset/seed${SEED}/prompts_batched.json \
    --template $TEMPLATE \
    --setting default \
    --output $OUT_PATH/seed$SEED
done
python get_result.py -m -p $OUT_PATH > $OUT_PATH/result.txt
done