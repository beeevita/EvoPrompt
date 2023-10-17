#!/bin/bash

set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8  
export CUDA_VISIBLE_DEVICES=0

BUDGET=10
POPSIZE=10
NUM_OF_MANUAL=10
SEED=5
GA=topk
TEMPLATE=v1

for dataset in sam
do
OUT_PATH=outputs/sum/$dataset/alpaca/all/ga/bd${BUDGET}_top${NUM_OF_MANUAL}_para_topk_init/$GA/$TEMPLATE/davinci
for SEED in 10 15
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
    --num_of_manual $NUM_OF_MANUAL \
    --random_data 0 \
    --position pre \
    --evo_mode ga \
    --llm_type davinci \
    --setting default \
    --initial all \
    --initial_mode para_topk \
    --ga_mode $GA \
    --cache_path data/sum/$dataset/seed${SEED}/prompts_batched.json \
    --template $TEMPLATE \
    --output $OUT_PATH/seed$SEED
done
python get_result.py -m -p $OUT_PATH > $OUT_PATH/result.txt
done